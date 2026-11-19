// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
// Sparse prefill attention that reads paged NVFP4 K/V directly.
//
// One CTA of four warps takes one query row; warp h computes KV group h: its
// twelve query heads (M=16 rows of mma.sync m16n8k16) against the row's
// selected keys, which all groups share. Per 32-key block the CTA gathers each
// selected token's 256-byte K and V rows (all four groups) and their scales
// once into a cp.async ring.
//
// Measured on B200 (32k queries x 2,184 keys): 64-key blocks lose (their
// 36 KB stages allow 2 CTAs per SM, not 3), and so does a dedicated producer
// warp (its registers cost the third CTA).
//
// NVFP4 values times their E4M3 scales are exact in FP16 (|x| <= 6 * 448), and
// so are E2M1 to FP16 conversions (one F2FP per pair) and the scale products
// (one HMUL2 per pair). Attention therefore runs FP16 MMAs with FP32
// accumulation: K and V are dequantized straight into B fragments, never
// into shared memory. Q is scaled by a power of two into FP16 range per warp
// (exact except for elements below 2^-29 of the row maximum); the softmax
// scale absorbs that power of two. P is rounded to FP16.
//
// The fragment layouts permute head dimensions consistently in Q and K, and
// output dimensions in V and O, so every thread loads contiguous bytes.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <torch/extension.h>

#include <cstdint>
#include <cstring>
#include <optional>

namespace {

constexpr int kPage = 128;
// Keys per online-softmax step and ring stage.
constexpr int kBlock = 32;
// Ring stages: one block is computed while the next one loads.
constexpr int kStages = 2;
// Q fragments: 8 k-steps of 16 bytes per lane.
constexpr int kQBytes = 8 * 32 * 16;

struct Params {
  const __nv_bfloat16* q;
  int64_t q_row;  // elements between query rows
  const uint8_t* storage;
  int64_t page_stride;  // bytes between pages
  const int* table;
  int64_t table_row;
  const int* requests;
  const int* route;
  int64_t route_row;
  const int* used;
  __nv_bfloat16* out;
  int64_t out_row;
  // Optional: process only rows of tiles flagged in `only_tiles` (rows of
  // `tile_rows` each), the rest having been computed by union attention.
  const int* only_tiles;
  int tile_rows;
  int rows;
  int width;
  float scale_log2;
};

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp16(uint32_t dst, const void* src, int bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst),
               "l"(src), "r"(bytes));
}

__device__ __forceinline__ void commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}

// Waits for every committed copy of this thread.
__device__ __forceinline__ void wait_copies() {
  asm volatile("cp.async.wait_group 0;\n" ::);
}

// The 32 bits of a packed pair (__half2, __nv_bfloat162 or their raw forms).
template <typename T>
__device__ __forceinline__ uint32_t bits(const T& pair) {
  static_assert(sizeof(T) == sizeof(uint32_t));
  uint32_t r;
  memcpy(&r, &pair, sizeof(r));
  return r;
}

__device__ __forceinline__ __half2 half2(uint32_t v) {
  __half2 r;
  memcpy(&r, &v, sizeof(r));
  return r;
}

// E2M1 pair in byte B of w to an FP16 pair (low nibble in the low half).
template <int B>
__device__ __forceinline__ uint32_t fp4x2(uint32_t w) {
  return bits(
      __nv_cvt_fp4x2_to_halfraw2(static_cast<uint8_t>(w >> (8 * B)), __NV_E2M1));
}

// Two E4M3 scales (low byte in the low half) to an FP16 pair.
__device__ __forceinline__ uint32_t fp8x2(uint32_t v) {
  return bits(__nv_cvt_fp8x2_to_halfraw2(static_cast<uint16_t>(v), __NV_E4M3));
}

__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {
  return bits(__hmul2_rn(half2(a), half2(b)));
}

__device__ __forceinline__ uint32_t pack_f16(float lo, float hi) {
  return bits(__floats2half2_rn(lo, hi));
}

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  return bits(__floats2bfloat162_rn(lo, hi));
}

// XXX: CUDA has no builtin for the approximate FP32 exp2 instruction.
__device__ __forceinline__ float exp2f_approx(float x) {
  float r;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}

__device__ __forceinline__ void mma(float* c, const uint32_t* a, uint32_t b0,
                                    uint32_t b1) {
  // Not volatile: independent MMAs may interleave across accumulators.
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint4 lds128(uint32_t a) {
  uint4 v;
  asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "r"(a));
  return v;
}

__device__ __forceinline__ uint2 lds64(uint32_t a) {
  uint2 v;
  asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "r"(a));
  return v;
}

__device__ __forceinline__ uint32_t lds16(uint32_t a) {
  uint16_t v;
  asm volatile("ld.shared.u16 %0, [%1];" : "=h"(v) : "r"(a));
  return v;
}

__device__ __forceinline__ uint32_t lds8(uint32_t a) {
  uint16_t v;
  asm volatile("ld.shared.u8 %0, [%1];" : "=h"(v) : "r"(a));
  return v;
}

// Stage of kBlock keys: K rows of all four KV groups (256 B each), V rows, then
// 32 bytes of K and of V scales per key. Swizzles depend on row % 8 only, so
// both the copies (16 lanes per 256-byte row) and the fragment reads below
// are free of bank conflicts.
struct Stage {
  static constexpr int kV = kBlock * 256;
  static constexpr int kKS = 2 * kBlock * 256;
  static constexpr int kVS = kKS + kBlock * 32;
  static constexpr int kBytes = kVS + kBlock * 32;
  // Rows 2m and 2m + 1 (read together) sit in opposite bank halves.
  static __device__ __forceinline__ uint32_t k(int r, int q) {
    return 256 * r + 16 * (q ^ ((r & 1) << 2));
  }
  // Rows 2t (t = 0..3, read together) take distinct 32-byte bank quarters.
  static __device__ __forceinline__ uint32_t v(int r, int q) {
    return kV + 256 * r + 16 * (q ^ (((r >> 1) & 3) << 1));
  }
  // Rows r and r + 4 swap 16-byte halves.
  static __device__ __forceinline__ uint32_t scale(int r, int byte) {
    return 32 * r + (byte ^ (((r >> 2) & 1) << 4));
  }
};

// One CTA of four warps per query row; warp h computes KV group h. At least
// three CTAs fit on an SM.
__global__ void __launch_bounds__(128, 3) nvfp4_attention_kernel(const Params p) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int tid = threadIdx.x;
  const int row = blockIdx.x;
  const int head = warp;
  if (p.only_tiles != nullptr && p.only_tiles[row / p.tile_rows] == 0) return;
  const int g = lane >> 2;
  const int t = lane & 3;
  const uint32_t ring = smem_addr(smem);
  // Q fragments live in shared memory, one 16-byte k-step per lane.
  const uint32_t qsm = ring + kStages * Stage::kBytes + warp * kQBytes + 16 * lane;

  const int used = min(p.used[row], p.width);
  const int blocks = (used + kBlock - 1) / kBlock;
  const int* table = p.table + int64_t(p.requests[row]) * p.table_row;
  const int* route = p.route + int64_t(row) * p.route_row;

  // Physical slot (page * 128 + token) of a key.
  auto locate = [&](int key) -> int {
    return __ldg(table + (key >> 7)) * kPage + (key & (kPage - 1));
  };
  // Caches exceed 64 GiB. Lanes share K/V row addresses as 32-bit counts of
  // 256-byte rows (page strides are multiples of 256), which covers 1 TiB.
  // Scale addresses use the 64-bit page base.
  const uint32_t page16 = static_cast<uint32_t>(p.page_stride / 16);
  const uint32_t page256 = static_cast<uint32_t>(p.page_stride / 256);
  const uint4* cache16 = reinterpret_cast<const uint4*>(p.storage);
  auto page_base = [&](uint32_t slot) -> const uint4* {
    return cache16 + uint64_t(slot >> 7) * page16;
  };
  // Lane l of every warp resolved key l of the block (`slot`, -1 past `used`);
  // 16 lanes copy each 256-byte row.
  auto issue = [&](int block, int slot) {
    const uint32_t stage = ring + (block % kStages) * Stage::kBytes;
    const uint32_t mine =
        slot < 0 ? ~0u
                 : (uint32_t(slot) >> 7) * page256 + (uint32_t(slot) & (kPage - 1));
    const int q = lane & 15;
#pragma unroll
    for (int j = 0; j < kBlock / 8; j++) {
      const int r = 8 * j + 2 * warp + (lane >> 4);
      const uint32_t shared = __shfl_sync(0xffffffffu, mine, r);
      const int bytes = shared != ~0u ? 16 : 0;
      const uint32_t x = shared != ~0u ? shared : 0u;
      const uint4* k = cache16 + uint64_t(x) * 16 + q;
      cp16(stage + Stage::k(r, q), k, bytes);
      cp16(stage + Stage::v(r, q), k + kPage * 16, bytes);
    }
    // Scales: per key 2 x 16 bytes of K scales (threads 0-63) and of V scales.
    const int r = (tid & 63) >> 1;
    const int half = tid & 1;
    const int is_v = tid >> 6;
    const uint32_t key = __shfl_sync(0xffffffffu, uint32_t(slot), r);
    const int bytes = int(key) >= 0 ? 16 : 0;
    const uint32_t kk = int(key) >= 0 ? key : 0u;
    const uint4* src = page_base(kk) + kPage * 32 + (kk & (kPage - 1)) * 2 + half +
                       is_v * kPage * 2;
    cp16(stage + (is_v ? Stage::kVS : Stage::kKS) + Stage::scale(r, 16 * half), src, bytes);
  };

  // Q fragments, scaled by 2^e into FP16 range. Thread (g, t) holds rows g and
  // g + 8, head dimensions 32t .. 32t + 31; k-step j covers 32t + 4j .. + 3.
  float scale_log2;
  {
    float qv[2][32];
    float amax = 0.f;
#pragma unroll
    for (int h = 0; h < 2; h++) {
      const int r = g + 8 * h;
      if (r < 12) {
        const uint4* src = reinterpret_cast<const uint4*>(
            p.q + int64_t(row) * p.q_row + (head * 12 + r) * 128 + 32 * t);
#pragma unroll
        for (int i = 0; i < 4; i++) {
          uint4 w = __ldg(src + i);
          uint32_t ws[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
          for (int k = 0; k < 4; k++) {
            qv[h][8 * i + 2 * k] = __uint_as_float(ws[k] << 16);
            qv[h][8 * i + 2 * k + 1] = __uint_as_float(ws[k] & 0xffff0000u);
          }
        }
      } else {
#pragma unroll
        for (int i = 0; i < 32; i++) qv[h][i] = 0.f;
      }
#pragma unroll
      for (int i = 0; i < 32; i++) amax = fmaxf(amax, fabsf(qv[h][i]));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    // amax in [2^E, 2^(E+1)) maps to [2^14, 2^15); FP16 max is 65504.
    int e = 0;
    if (amax > 0.f) e = 141 - int((__float_as_uint(amax) >> 23) & 0xff);
    e = max(-100, min(100, e));
    const float up = __uint_as_float(uint32_t(127 + e) << 23);
    const float down = __uint_as_float(uint32_t(127 - e) << 23);
    scale_log2 = p.scale_log2 * down;
#pragma unroll
    for (int j = 0; j < 8; j++) {
      const uint32_t a0 = pack_f16(qv[0][4 * j] * up, qv[0][4 * j + 1] * up);
      const uint32_t a1 = pack_f16(qv[1][4 * j] * up, qv[1][4 * j + 1] * up);
      const uint32_t a2 = pack_f16(qv[0][4 * j + 2] * up, qv[0][4 * j + 3] * up);
      const uint32_t a3 = pack_f16(qv[1][4 * j + 2] * up, qv[1][4 * j + 3] * up);
      asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(qsm + 512 * j),
                   "r"(a0), "r"(a1), "r"(a2), "r"(a3));
    }
  }

  // Ring prologue: block 0 issued, keys of block 1 resolved, of block 2 loaded.
  int next_slot, next_key;
  {
    const int slot = lane < used ? locate(__ldg(route + lane)) : -1;
    if (blocks > 0) issue(0, slot);
    commit();
    const int pos = kBlock + lane;
    next_slot = pos < used ? locate(__ldg(route + pos)) : -1;
    next_key = pos + kBlock < used ? __ldg(route + pos + kBlock) : -1;
  }

  float o[16][4];
#pragma unroll
  for (int c = 0; c < 16; c++)
#pragma unroll
    for (int i = 0; i < 4; i++) o[c][i] = 0.f;
  float row_max[2] = {-INFINITY, -INFINITY};
  float row_sum[2] = {0.f, 0.f};

  for (int b = 0; b < blocks; b++) {
    // Block b has landed for every thread, and every warp is done with the
    // stage the refill below overwrites.
    wait_copies();
    __syncthreads();
    {
      if (b + 1 < blocks) issue(b + 1, next_slot);
      commit();
      next_slot = next_key >= 0 ? locate(next_key) : -1;
      const int pos = (b + 3) * kBlock + lane;
      next_key = pos < used ? __ldg(route + pos) : -1;
    }
    const uint32_t stage = ring + (b % kStages) * Stage::kBytes;

    // S = Q K^T over four tiles of 8 keys; tile i holds keys 8i + g. The
    // K-steps run outermost so consecutive MMAs use independent accumulators.
    float s[kBlock / 8][4];
    uint4 kw[4];
    uint32_t sc[4][2];
#pragma unroll
    for (int i = 0; i < 4; i++) {
      s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.f;
      const int r = 8 * i + g;
      kw[i] = lds128(stage + Stage::k(r, 4 * head + t));
      const uint32_t pair =
          fp8x2(lds16(stage + Stage::kKS + Stage::scale(r, 8 * head + 2 * t)));
      sc[i][0] = __byte_perm(pair, pair, 0x1010);
      sc[i][1] = __byte_perm(pair, pair, 0x3232);
    }
#pragma unroll
    for (int j = 0; j < 8; j++) {
      const uint4 qj = lds128(qsm + 512 * j);
      const uint32_t qa[4] = {qj.x, qj.y, qj.z, qj.w};
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const uint32_t w[4] = {kw[i].x, kw[i].y, kw[i].z, kw[i].w};
        const uint32_t word = w[j >> 1];
        const uint32_t scale = sc[i][j >> 2];
        uint32_t b0, b1;
        if ((j & 1) == 0) {
          b0 = fp4x2<0>(word);
          b1 = fp4x2<1>(word);
        } else {
          b0 = fp4x2<2>(word);
          b1 = fp4x2<3>(word);
        }
        mma(s[i], qa, hmul2(b0, scale), hmul2(b1, scale));
      }
    }

    // Online softmax; thread (g, t) holds keys 8i + 2t + {0, 1} of rows g, g+8.
    if ((b + 1) * kBlock > used) {
#pragma unroll
      for (int i = 0; i < kBlock / 8; i++) {
        const int key = b * kBlock + 8 * i + 2 * t;
        if (key >= used) s[i][0] = s[i][2] = -INFINITY;
        if (key + 1 >= used) s[i][1] = s[i][3] = -INFINITY;
      }
    }
#pragma unroll
    for (int h = 0; h < 2; h++) {
      float mx = row_max[h];
#pragma unroll
      for (int i = 0; i < kBlock / 8; i++)
        mx = fmaxf(mx, fmaxf(s[i][2 * h], s[i][2 * h + 1]));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
      const float safe = mx == -INFINITY ? 0.f : mx;
      const float alpha = exp2f_approx((row_max[h] - safe) * scale_log2);
      row_max[h] = mx;
      const float offset = safe * scale_log2;
      float sum = 0.f;
#pragma unroll
      for (int i = 0; i < kBlock / 8; i++) {
        const float p0 = exp2f_approx(s[i][2 * h] * scale_log2 - offset);
        const float p1 = exp2f_approx(s[i][2 * h + 1] * scale_log2 - offset);
        sum += p0 + p1;
        s[i][2 * h] = p0;
        s[i][2 * h + 1] = p1;
      }
      row_sum[h] = row_sum[h] * alpha + sum;
      // Scaling by 1 is exact; skip it unless some row's maximum moved.
      if (__any_sync(0xffffffffu, alpha != 1.f)) {
#pragma unroll
        for (int c = 0; c < 16; c++) {
          o[c][2 * h] *= alpha;
          o[c][2 * h + 1] *= alpha;
        }
      }
    }

    // O += P V. Thread (g, t) holds output dims 16g .. 16g + 15 of each V row;
    // n-tile c is dim 16g + c of every thread group g.
#pragma unroll
    for (int k = 0; k < kBlock / 16; k++) {
      const uint32_t pa[4] = {
          pack_f16(s[2 * k][0], s[2 * k][1]),
          pack_f16(s[2 * k][2], s[2 * k][3]),
          pack_f16(s[2 * k + 1][0], s[2 * k + 1][1]),
          pack_f16(s[2 * k + 1][2], s[2 * k + 1][3]),
      };
      uint32_t frag[2][16];
#pragma unroll
      for (int h = 0; h < 2; h++) {
        const int ra = 16 * k + 2 * t + 8 * h;
        const int rb = ra + 1;
        const int q = 4 * head + (g >> 1);
        const uint2 wa = lds64(stage + Stage::v(ra, q) + 8 * (g & 1));
        const uint2 wb = lds64(stage + Stage::v(rb, q) + 8 * (g & 1));
        const uint32_t sa = lds8(stage + Stage::kVS + Stage::scale(ra, 8 * head + g));
        const uint32_t sb = lds8(stage + Stage::kVS + Stage::scale(rb, 8 * head + g));
        const uint32_t scale = fp8x2(sa | (sb << 8));
        const uint32_t e[4] = {
            (wa.x & 0x0f0f0f0fu) | ((wb.x << 4) & 0xf0f0f0f0u),
            ((wa.x >> 4) & 0x0f0f0f0fu) | (wb.x & 0xf0f0f0f0u),
            (wa.y & 0x0f0f0f0fu) | ((wb.y << 4) & 0xf0f0f0f0u),
            ((wa.y >> 4) & 0x0f0f0f0fu) | (wb.y & 0xf0f0f0f0u),
        };
        // Byte m of e[x] holds dim 2m + (x & 1) + 8 (x >> 1).
#pragma unroll
        for (int x = 0; x < 4; x++) {
          const int base = (x & 1) + 8 * (x >> 1);
          frag[h][base + 0] = hmul2(fp4x2<0>(e[x]), scale);
          frag[h][base + 2] = hmul2(fp4x2<1>(e[x]), scale);
          frag[h][base + 4] = hmul2(fp4x2<2>(e[x]), scale);
          frag[h][base + 6] = hmul2(fp4x2<3>(e[x]), scale);
        }
      }
#pragma unroll
      for (int c = 0; c < 16; c++) mma(o[c], pa, frag[0][c], frag[1][c]);
    }
  }
  wait_copies();

  // Rows g and g + 8; thread (g, t) holds dims 32t + c (n = 2t) and
  // 32t + 16 + c (n = 2t + 1) of n-tile c.
#pragma unroll
  for (int h = 0; h < 2; h++) {
    float sum = row_sum[h];
    sum += __shfl_xor_sync(0xffffffffu, sum, 1);
    sum += __shfl_xor_sync(0xffffffffu, sum, 2);
    const int r = g + 8 * h;
    if (r >= 12) continue;
    const float inv = sum > 0.f ? 1.f / sum : 0.f;
    uint32_t packed[16];
#pragma unroll
    for (int c = 0; c < 16; c += 2) {
      packed[c / 2] = pack_bf16(o[c][2 * h] * inv, o[c + 1][2 * h] * inv);
      packed[8 + c / 2] =
          pack_bf16(o[c][2 * h + 1] * inv, o[c + 1][2 * h + 1] * inv);
    }
    uint4* dst = reinterpret_cast<uint4*>(
        p.out + int64_t(row) * p.out_row + (head * 12 + r) * 128 + 32 * t);
#pragma unroll
    for (int i = 0; i < 4; i++)
      dst[i] = make_uint4(packed[4 * i], packed[4 * i + 1], packed[4 * i + 2],
                          packed[4 * i + 3]);
  }
}

void launch(const Params& p, cudaStream_t stream) {
  constexpr int smem = kStages * Stage::kBytes + 4 * kQBytes;
  // The attribute is per device; set it once for each.
  static bool configured[64] = {};
  int device = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  if (device >= 64 || !configured[device]) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        nvfp4_attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    if (device < 64) configured[device] = true;
  }
  nvfp4_attention_kernel<<<p.rows, 128, smem, stream>>>(p);
}

}  // namespace

// q, out: [rows, 48, 128] BF16 (any row stride); storage: [pages, bytes] uint8
// NVFP4 pages of 128 tokens; table: [requests, pages] int32; requests, used:
// [rows] int32; route: [rows, width] int32 logical key positions.
//
// Pre-conditions that are not checked: the data of q, out and storage starts
// at a 16-byte aligned address; route[row, :used[row]] holds positions whose
// pages table[requests[row]] maps to pages of storage.
void nvfp4_attention(torch::Tensor q, torch::Tensor storage, torch::Tensor table,
                     torch::Tensor requests, torch::Tensor route, torch::Tensor used,
                     torch::Tensor out, std::optional<torch::Tensor> only_tiles, int64_t tile_rows,
                     double softmax_scale) {
  TORCH_CHECK(q.scalar_type() == at::kBFloat16 && out.scalar_type() == at::kBFloat16);
  TORCH_CHECK(storage.scalar_type() == at::kByte && storage.stride(1) == 1);
  TORCH_CHECK(table.scalar_type() == at::kInt && table.stride(1) == 1);
  TORCH_CHECK(route.scalar_type() == at::kInt && route.stride(1) == 1);
  TORCH_CHECK(requests.scalar_type() == at::kInt && requests.is_contiguous());
  TORCH_CHECK(used.scalar_type() == at::kInt && used.is_contiguous());
  TORCH_CHECK(q.dim() == 3 && q.size(1) == 48 && q.size(2) == 128);
  TORCH_CHECK(q.stride(2) == 1 && q.stride(1) == 128 && q.stride(0) % 8 == 0);
  TORCH_CHECK(out.sizes() == q.sizes() && out.stride(2) == 1 && out.stride(1) == 128);
  TORCH_CHECK(out.stride(0) % 8 == 0);
  TORCH_CHECK(storage.stride(0) % 256 == 0 && storage.size(1) >= kPage * 576);
  // Slots (page * 128 + token), page strides in 16-byte units and counts of
  // 256-byte rows are 32-bit; the largest row count marks a missing key.
  TORCH_CHECK(storage.size(0) <= (int64_t(1) << 24));
  TORCH_CHECK(storage.stride(0) / 16 < (int64_t(1) << 32));
  TORCH_CHECK(storage.size(0) * storage.stride(0) / 256 < (int64_t(1) << 32) - 1);
  const int64_t rows = q.size(0);
  TORCH_CHECK(route.size(0) == rows && used.size(0) == rows && requests.size(0) == rows);
  if (rows == 0) return;
  TORCH_CHECK(rows < int64_t(1) << 31);
  const c10::cuda::CUDAGuard guard(q.device());
  Params p{};
  p.q = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr());
  p.q_row = q.stride(0);
  p.storage = storage.data_ptr<uint8_t>();
  p.page_stride = storage.stride(0);
  p.table = table.data_ptr<int>();
  p.table_row = table.stride(0);
  p.requests = requests.data_ptr<int>();
  p.route = route.data_ptr<int>();
  p.route_row = route.stride(0);
  p.used = used.data_ptr<int>();
  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.out_row = out.stride(0);
  if (only_tiles.has_value()) {
    TORCH_CHECK(only_tiles->scalar_type() == at::kInt && only_tiles->is_contiguous());
    TORCH_CHECK(tile_rows > 0 && only_tiles->size(0) * tile_rows >= rows);
    p.only_tiles = only_tiles->data_ptr<int>();
    p.tile_rows = static_cast<int>(tile_rows);
  }
  p.rows = static_cast<int>(rows);
  p.width = static_cast<int>(route.size(1));
  p.scale_log2 = static_cast<float>(softmax_scale * 1.4426950408889634);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  launch(p, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nvfp4_attention", &nvfp4_attention);
}
