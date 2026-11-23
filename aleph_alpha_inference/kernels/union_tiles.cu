// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
// Union tiles for sparse prefill attention on FA4.
//
// Consecutive query rows of one request form tiles of T rows. Per tile, the
// kernel writes the sorted union of the rows' selected keys (logical
// positions, FA4's page table at page size 1), its length, and per row a
// bitmask over the union (bit u of row q: union[u] is one of q's keys), the
// layout vLLM's dense_mask_mod reads.
//
// One CTA per tile: the keys are set in a shared-memory bitmap over the
// request's span; popcount prefix sums rank them; each warp then builds one
// row's mask in shared memory and copies it out.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

constexpr int kThreads = 512;
constexpr int kWarps = kThreads / 32;

// Calls visit(q, key) for every selected key of the tile's nq rows. Loads are
// independent and batched (kUnroll per thread) to hide latency.
template <class Visit>
__device__ __forceinline__ void for_each_key(const int* route, int64_t route_row,
                                             const int* used, int first, int nq,
                                             int width, int tid, Visit visit) {
  constexpr int kUnroll = 8;
  for (int q = 0; q < nq; q++) {
    const int n = min(used[first + q], width);
    const int* r = route + int64_t(first + q) * route_row;
    for (int base = 0; base < n; base += kThreads * kUnroll) {
      int key[kUnroll];
#pragma unroll
      for (int u = 0; u < kUnroll; u++) {
        const int i = base + u * kThreads + tid;
        key[u] = i < n ? __ldg(r + i) : -1;
      }
#pragma unroll
      for (int u = 0; u < kUnroll; u++)
        if (key[u] >= 0) visit(q, key[u]);
    }
  }
}

__global__ void __launch_bounds__(kThreads) union_tiles_kernel(
    const int* __restrict__ route, int64_t route_row, const int* __restrict__ used,
    int rows, int width, int tile_rows, int span_words, int limit,
    int* __restrict__ keys, int64_t keys_row, int* __restrict__ count,
    int* __restrict__ fallback, int* __restrict__ qrows, uint32_t* __restrict__ mask,
    int64_t mask_tile, int64_t mask_row, int* __restrict__ union_sum) {
  extern __shared__ __align__(16) uint32_t smem[];
  uint32_t* bitmap = smem;  // span_words
  // Rank of each bitmap word's first key.
  uint16_t* prefix = reinterpret_cast<uint16_t*>(bitmap + span_words);  // span_words
  const int mask_words = static_cast<int>(mask_row);
  // Rows selecting each union key (bit q: row q), one word per union rank.
  uint32_t* rows_of = bitmap + span_words + (span_words + 1) / 2;  // mask_words * 32
  __shared__ int warp_sums[kWarps];
  __shared__ int total;

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int tile = blockIdx.x;
  const int first = tile * tile_rows;
  const int nq = min(tile_rows, rows - first);
  for (int w = tid; w < span_words; w += kThreads) bitmap[w] = 0;
  __syncthreads();
  for_each_key(route, route_row, used, first, nq, width, tid, [&](int, int key) {
    // Most keys repeat across a tile's rows: skip the atomic when set.
    if (!(bitmap[key >> 5] & (1u << (key & 31))))
      atomicOr(bitmap + (key >> 5), 1u << (key & 31));
  });
  __syncthreads();

  // Exclusive popcount prefix over the bitmap, contiguous chunks per thread.
  const int chunk = (span_words + kThreads - 1) / kThreads;
  const int lo = min(tid * chunk, span_words), hi = min(lo + chunk, span_words);
  int sum = 0;
  for (int w = lo; w < hi; w++) sum += __popc(bitmap[w]);
  int incl = sum;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int x = __shfl_up_sync(0xffffffffu, incl, o);
    if (lane >= o) incl += x;
  }
  if (lane == 31) warp_sums[warp] = incl;
  __syncthreads();
  if (warp == 0) {
    int s = lane < kWarps ? warp_sums[lane] : 0;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int x = __shfl_up_sync(0xffffffffu, s, o);
      if (lane >= o) s += x;
    }
    if (lane < kWarps) warp_sums[lane] = s;
    if (lane == kWarps - 1) total = s;
  }
  __syncthreads();
  int run = incl - sum + (warp > 0 ? warp_sums[warp - 1] : 0);
  int* out_keys = keys + int64_t(tile) * keys_row;
  const int union_size = total;
  // Tiles whose union exceeds `limit` go to the per-row kernel instead.
  const bool spill = union_size > limit;
  for (int w = lo; w < hi; w++) {
    uint32_t bits = bitmap[w];
    prefix[w] = static_cast<uint16_t>(min(run, 65535));
    int pos = run;
    while (bits) {
      const int b = __ffs(bits) - 1;
      if (!spill) out_keys[pos] = w * 32 + b;
      pos++;
      bits &= bits - 1;
    }
    run = pos;
  }
  if (tid == 0) {
    count[tile] = spill ? 0 : union_size;
    fallback[tile] = spill;
    qrows[tile] = spill ? 0 : nq;
    atomicAdd(union_sum, union_size);
  }
  if (spill) return;
  __syncthreads();  // every word's prefix is written

  // Row masks. First each (row, key) sets bit `row` of its key's rank word:
  // a row's keys are distinct and consecutive keys have consecutive ranks, so
  // these atomics rarely collide. Then each warp transposes 32 ranks at a time
  // with one ballot per row.
  for (int u = tid; u < union_size; u += kThreads) rows_of[u] = 0;
  __syncthreads();
  for_each_key(route, route_row, used, first, nq, width, tid, [&](int q, int key) {
    const int w = key >> 5;
    const int rank = prefix[w] + __popc(bitmap[w] & ((1u << (key & 31)) - 1u));
    atomicOr(rows_of + rank, 1u << q);
  });
  __syncthreads();
  const int live_words = min(mask_words, ((union_size + 127) / 128) * 4);
  uint32_t* dst = mask + int64_t(tile) * mask_tile;
  for (int w = warp; w < live_words; w += kWarps) {
    const int u = w * 32 + lane;
    const uint32_t rows = u < union_size ? rows_of[u] : 0u;
    uint32_t mine = 0;
    for (int q = 0; q < nq; q++) {
      const uint32_t bits = __ballot_sync(0xffffffffu, (rows >> q) & 1u);
      if (lane == q) mine = bits;
    }
    if (lane < nq) dst[int64_t(lane) * mask_row + w] = mine;
  }
}

// Requests whose tiles average more than `gate` union keys run entirely on
// the per-row kernel: FA4 costs grow with the union, and a few remaining
// tiles would leave FA4 a long tail.
__global__ void gate_tiles_kernel(int tiles, int gate, const int* __restrict__ union_sum,
                                  int* __restrict__ count, int* __restrict__ fallback,
                                  int* __restrict__ qrows) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t < tiles && int64_t(*union_sum) > int64_t(gate) * tiles) {
    count[t] = 0;
    fallback[t] = 1;
    qrows[t] = 0;
  }
}

constexpr int64_t kMaxSmem = 227 * 1024;

// Shared memory of a tile for keys in [0, span) and unions of up to
// mask_words * 32 keys: bitmap, rank prefix (uint16 per word) and per-rank
// row words.
int64_t union_smem(int64_t span, int64_t mask_words) {
  const int64_t span_words = (span + 31) / 32;
  return span_words * 4 + (span_words + 1) / 2 * 4 + mask_words * 32 * 4;
}

}  // namespace

// route: [rows, width] int32 logical key positions of one request, used:
// [rows]; keys: [tiles, cap] int32; count: [tiles]; mask: [tiles, >=tile_rows,
// words] int32 with words * 32 >= cap. `span` bounds every key.
void union_tiles(torch::Tensor route, torch::Tensor used, int64_t tile_rows, int64_t span,
                 int64_t limit, int64_t gate, torch::Tensor keys, torch::Tensor count,
                 torch::Tensor fallback, torch::Tensor qrows, torch::Tensor mask,
                 torch::Tensor union_sum) {
  TORCH_CHECK(route.scalar_type() == at::kInt && route.stride(1) == 1);
  TORCH_CHECK(used.scalar_type() == at::kInt && used.is_contiguous());
  TORCH_CHECK(keys.scalar_type() == at::kInt && keys.stride(1) == 1);
  TORCH_CHECK(count.scalar_type() == at::kInt && count.is_contiguous());
  TORCH_CHECK(mask.scalar_type() == at::kInt && mask.dim() == 3 && mask.stride(2) == 1);
  const int64_t rows = route.size(0);
  const int64_t tiles = (rows + tile_rows - 1) / tile_rows;
  TORCH_CHECK(keys.size(0) >= tiles && count.size(0) >= tiles && mask.size(0) >= tiles);
  TORCH_CHECK(mask.size(1) >= tile_rows && mask.stride(1) == mask.size(2));
  TORCH_CHECK(tile_rows <= 32, "row bits are one word");
  TORCH_CHECK(fallback.scalar_type() == at::kInt && fallback.size(0) >= tiles);
  TORCH_CHECK(qrows.scalar_type() == at::kInt && qrows.size(0) >= tiles);
  TORCH_CHECK(0 < limit && limit <= keys.size(1) && limit < 65536);
  TORCH_CHECK(mask.size(2) * 32 >= limit && mask.size(2) % 4 == 0);
  if (tiles == 0) return;
  const int span_words = static_cast<int>((span + 31) / 32);
  const int64_t smem = union_smem(span, mask.size(2));
  TORCH_CHECK(smem <= kMaxSmem, "union tile span exceeds shared memory");
  TORCH_CHECK(union_sum.scalar_type() == at::kInt && union_sum.numel() >= 1);
  const c10::cuda::CUDAGuard guard(route.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaMemsetAsync(union_sum.data_ptr<int>(), 0, sizeof(int), stream));
  C10_CUDA_CHECK(cudaFuncSetAttribute(union_tiles_kernel,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      static_cast<int>(smem)));
  union_tiles_kernel<<<tiles, kThreads, smem, stream>>>(
      route.data_ptr<int>(), route.stride(0), used.data_ptr<int>(), static_cast<int>(rows),
      static_cast<int>(route.size(1)), static_cast<int>(tile_rows), span_words,
      static_cast<int>(limit), keys.data_ptr<int>(), keys.stride(0), count.data_ptr<int>(),
      fallback.data_ptr<int>(), qrows.data_ptr<int>(),
      reinterpret_cast<uint32_t*>(mask.data_ptr<int>()), mask.stride(0), mask.stride(1),
      union_sum.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  gate_tiles_kernel<<<(tiles + 255) / 256, 256, 0, stream>>>(
      static_cast<int>(tiles), static_cast<int>(gate), union_sum.data_ptr<int>(),
      count.data_ptr<int>(), fallback.data_ptr<int>(), qrows.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("union_tiles", &union_tiles);
}
