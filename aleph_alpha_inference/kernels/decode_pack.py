# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""CuTe NVFP4 query packing for routing."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int64

from aleph_alpha_inference.kernels.cute_jit import Executables
from aleph_alpha_inference.kernels.fp16_fragments import fp4_pair


class PackQuery:
    def __init__(self, heads=12):
        self.heads = heads

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        packed: cute.Tensor,
        scales: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(q, packed, scales).launch(
            grid=(cute.ceil_div(q.shape[0] * self.heads, 4), 1, 1),
            block=(128, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        q: cute.Tensor,
        packed: cute.Tensor,
        scales: cute.Tensor,
    ):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        vector = bid * 4 + tid // 32
        if vector < q.shape[0] * self.heads:
            row = vector // self.heads
            head = vector % self.heads
            lane = tid % 32
            # Each lane converts four adjacent values. Four lanes share one
            # 16-value scale block.
            pos = lane * 4
            block = lane // 4
            source = cute.make_tensor(
                q.iterator + Int64(row) * q.stride[0] + head * q.stride[1] + pos,
                cute.make_layout((4,)),
            )
            values = source.load().to(Float32)
            # NaNs count as zero, so they do not set the block's scale.
            magnitudes = cute.where(values == values, abs(values), Float32(0))
            peak = Float32(magnitudes.reduce(cute.ReductionOp.MAX, Float32(0), 0))
            peak = cute.arch.warp_reduction_max(peak, threads_in_group=4)
            scale = cutlass.Float8E4M3FN(
                cutlass.max(peak / Float32(6), Float32(0.001953125))
            )
            # Division preserves nearest-even ties; reciprocal multiplication does not.
            scaled = values / Float32(scale)
            offset = Int64(row) * packed.stride[0] + head * packed.stride[1] + pos // 2
            destination = cute.make_tensor(
                packed.iterator + offset, cute.make_layout((2,))
            )
            destination[0] = fp4_pair(scaled[0], scaled[1])
            destination[1] = fp4_pair(scaled[2], scaled[3])
            if lane % 4 == 0:
                scales[Int64(row), head, block] = scale


_COMPILED = Executables("decode_pack")


class DecodeQueryPack:
    """Pack BF16 queries into NVFP4 routing queries with one FP8 scale per
    16-value block."""

    def __call__(self, q, packed, scales):
        tensors = [
            q.view(q.shape[0], 48, 128),
            packed.view(torch.uint8),
            scales.view(torch.float8_e4m3fn),
        ]
        aligns = [2 if t.dtype == torch.bfloat16 else 4 for t in tensors]
        _COMPILED.launch(PackQuery(), tensors, aligns=aligns)
