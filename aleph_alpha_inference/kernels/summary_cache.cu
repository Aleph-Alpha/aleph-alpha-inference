// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <torch/extension.h>
// One launch per append: each warp quantizes one token's K/V (and head 0's
// routing key) with one 16-value scale group per lane, then the warp at the
// start of each contiguous segment within a summary block updates its mean.
// A page has one partial sum. When one append both completes a block that an
// earlier append opened (reading the old sum) and leaves a later block of the
// same page open (writing the new sum), the completing warp does both, in
// that order; the open segment's own warp skips it.
//
// Pre-condition: the rows of one append that write to the same page hold
// consecutive, increasing slots.

// Tokens per page and per summary block.
constexpr int kPage = 128;
constexpr int kBlock = 16;
// The smallest scale: 2^-9, the smallest positive E4M3 value.
constexpr float kMinScale = 0.001953125f;

// 16 values as NVFP4: eight E2M1 pairs and their E4M3 scale.
struct Group {
  uint2 packed;
  unsigned char scale;
};

__device__ __forceinline__ Group quantize16(const nv_bfloat16 *src) {
  uint4 raw[2];
  raw[0] = reinterpret_cast<const uint4 *>(src)[0];
  raw[1] = reinterpret_cast<const uint4 *>(src)[1];
  const nv_bfloat16 *x = reinterpret_cast<const nv_bfloat16 *>(raw);
  float v[16], amax = 0.f;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    v[i] = __bfloat162float(x[i]);
    amax = fmaxf(amax, fabsf(v[i]));
  }
  Group group;
  group.scale = __nv_cvt_float_to_fp8(fmaxf(amax / 6.f, kMinScale),
                                      __NV_SATFINITE, __NV_E4M3);
  float s = __half2float(__nv_cvt_fp8_to_halfraw(group.scale, __NV_E4M3));
  unsigned char *bytes = reinterpret_cast<unsigned char *>(&group.packed);
#pragma unroll
  for (int i = 0; i < 8; ++i)
    bytes[i] = __nv_cvt_float2_to_fp4x2(
        make_float2(v[2 * i] / s, v[2 * i + 1] / s), __NV_E2M1,
        cudaRoundNearest);
  return group;
}

__device__ __forceinline__ void store(const Group &group, unsigned char *packed,
                                      unsigned char *scale) {
  *reinterpret_cast<uint2 *>(packed) = group.packed;
  *scale = group.scale;
}

__device__ __forceinline__ bool follows(const int64_t *slots, int r,
                                        int64_t slot) {
  return r >= 0 && slots[r] == slot;
}

// Sum head 0's keys of rows [r, r + count) into vals, starting from the
// page's partial sum when the block was already open.
__device__ __forceinline__ void segment_sum(
    const nv_bfloat16 *k, int64_t k_stride, int r, int count, bool open,
    const float *partial, float *vals, int lane) {
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    int d = lane + j * 32;
    float sum = open ? partial[d] : 0.f;
    for (int t = 0; t < count; ++t)
      sum += __bfloat162float(k[int64_t(r + t) * k_stride + d]);
    vals[j] = sum;
  }
}

// Quantize the block mean of the sums in vals and store it with its scales.
__device__ __forceinline__ void store_mean(float *vals, unsigned char *mean,
                                           unsigned char *scales, int lane) {
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    float value = vals[j] / kBlock, a = fabsf(value);
    for (int delta = 8; delta; delta /= 2)
      a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, delta));
    unsigned char sb = __nv_cvt_float_to_fp8(fmaxf(a / 6.f, kMinScale),
                                             __NV_SATFINITE, __NV_E4M3);
    float scale = __half2float(__nv_cvt_fp8_to_halfraw(sb, __NV_E4M3));
    float v = value / scale, next = __shfl_down_sync(0xffffffff, v, 1);
    if (!(lane & 1))
      mean[j * 16 + lane / 2] = __nv_cvt_float2_to_fp4x2(
          make_float2(v, next), __NV_E2M1, cudaRoundNearest);
    if (lane % 16 == 0)
      scales[j * 2 + lane / 16] = sb;
  }
}

__global__ void append_summary_kernel(
    const nv_bfloat16 *k, const nv_bfloat16 *v, const int64_t *slots, int rows,
    int64_t k_stride, int64_t v_stride, unsigned char *storage, int64_t stride,
    unsigned char *routing, int64_t routing_stride, unsigned char *mean,
    unsigned char *scales, float *partial, int64_t mean_stride,
    int64_t scale_stride, int64_t partial_stride) {
  constexpr int page = kPage, block = kBlock;
  int r = blockIdx.x * 4 + threadIdx.x / 32, lane = threadIdx.x % 32;
  if (r >= rows)
    return;
  int64_t slot = slots[r];
  if (slot < 0)
    return;
  int64_t p = slot / page;
  int offset = slot % page, head = lane / 8, group = lane % 8;
  auto *base = storage + p * stride;
  int dest = offset * 4 + head;
  const Group key = quantize16(k + int64_t(r) * k_stride + lane * 16);
  store(key, base + dest * 64 + group * 8, base + 512 * page + dest * 8 + group);
  store(quantize16(v + int64_t(r) * v_stride + lane * 16),
        base + 256 * page + dest * 64 + group * 8,
        base + 544 * page + dest * 8 + group);
  if (lane < 8) {
    // Head 0's key is also the routing key.
    auto *rb = routing + p * routing_stride;
    store(key, rb + offset * 64 + lane * 8, rb + 64 * page + offset * 8 + lane);
  }
  // Summary: only the first row of a contiguous segment within a block.
  int in_block = slot % block;
  if (in_block && follows(slots, r - 1, slot - 1))
    return;
  auto count_from = [&](int row, int64_t first) {
    int n = 1, limit = block - int(first % block);
    while (n < limit && row + n < rows && slots[row + n] == first + n)
      ++n;
    return n;
  };
  int count = count_from(r, slot);
  bool complete = in_block + count == block;
  float *page_partial = partial + p * partial_stride;
  int64_t page_start = p * page;
  float vals[4];
  if (!complete) {
    // An open segment starting a block: skip if a completing segment of this
    // append opened earlier in the page (that warp writes this sum).
    if (!in_block) {
      int64_t s = slot;
      int row = r;
      while (s > page_start && follows(slots, row - 1, s - 1)) {
        --s;
        --row;
      }
      if (s % block)
        return;
    }
    segment_sum(k, k_stride, r, count, in_block, page_partial, vals, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j)
      page_partial[lane + j * 32] = vals[j];
    return;
  }
  segment_sum(k, k_stride, r, count, in_block, page_partial, vals, lane);
  int b = offset / block;
  store_mean(vals, mean + p * mean_stride + b * 64,
             scales + p * scale_stride + b * 8, lane);
  if (!in_block)
    return;
  // This segment read the old sum; find this append's open tail in the page.
  int row = r + count;
  int64_t s = slot + count;
  while (s < page_start + page && row < rows && slots[row] == s &&
         s % block == 0) {
    int n = count_from(row, s);
    if (n < block) {
      float tail[4];
      segment_sum(k, k_stride, row, n, false, page_partial, tail, lane);
#pragma unroll
      for (int j = 0; j < 4; ++j)
        page_partial[lane + j * 32] = tail[j];
      return;
    }
    row += n;
    s += n;
  }
}

// k, v: [rows, 4, 128] BF16; slots: [rows] int64 token slots (page * 128 +
// offset), negative ones are skipped; storage, routing: [pages, bytes] uint8;
// mean: [pages, 8, 64] uint8; scales: [pages, 8, 2] int32; partial:
// [pages, 128] FP32.
void append_summary(torch::Tensor k, torch::Tensor v, torch::Tensor slots,
                    torch::Tensor storage, torch::Tensor routing,
                    torch::Tensor mean, torch::Tensor scales,
                    torch::Tensor partial) {
  c10::cuda::CUDAGuard g(k.device());
  TORCH_CHECK(k.scalar_type() == at::kBFloat16 &&
              v.scalar_type() == at::kBFloat16 && k.dim() == 3 &&
              k.size(1) == 4 && k.size(2) == 128 && v.sizes() == k.sizes() &&
              k.stride(1) == 128 && k.stride(2) == 1 && v.stride(1) == 128 &&
              v.stride(2) == 1);
  TORCH_CHECK(k.stride(0) % 16 == 0 && v.stride(0) % 16 == 0 &&
              reinterpret_cast<uintptr_t>(k.data_ptr()) % 32 == 0 &&
              reinterpret_cast<uintptr_t>(v.data_ptr()) % 32 == 0,
              "append_summary requires 32-byte aligned K/V rows");
  TORCH_CHECK(storage.scalar_type() == at::kByte && storage.stride(1) == 1 &&
              storage.size(1) >= kPage * 576 && storage.stride(0) % 8 == 0 &&
              routing.scalar_type() == at::kByte && routing.stride(1) == 1 &&
              routing.size(1) >= kPage * 72 && routing.stride(0) % 8 == 0 &&
              routing.size(0) == storage.size(0));
  TORCH_CHECK(slots.scalar_type() == at::kLong && slots.is_contiguous() &&
              slots.numel() == k.size(0));
  int rows = slots.numel(), grid = (rows + 3) / 4;
  if (!grid)
    return;
  auto s = at::cuda::getCurrentCUDAStream();
  append_summary_kernel<<<grid, 128, 0, s>>>(
      (nv_bfloat16 *)k.data_ptr(), (nv_bfloat16 *)v.data_ptr(),
      slots.data_ptr<int64_t>(), rows, k.stride(0), v.stride(0),
      storage.data_ptr<unsigned char>(), storage.stride(0),
      routing.data_ptr<unsigned char>(), routing.stride(0),
      mean.data_ptr<unsigned char>(), (unsigned char *)scales.data_ptr(),
      partial.data_ptr<float>(), mean.stride(0), scales.stride(0) * 4,
      partial.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("append_summary", &append_summary);
}
