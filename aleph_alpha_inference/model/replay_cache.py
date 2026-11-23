# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""Clear the unwritten keys below a replay start in its aligned FA4 tile."""

from math import lcm

import cuda.bindings.driver as cuda
import cutlass.cute as cute
from cutlass import Int64

from aleph_alpha_inference.kernels.cute_jit import Executables


def replay_cache_alignment(page):
    """Tokens of the block-table origin a replay rebase keeps: whole pages and
    whole FA4 SM100 D128 128-key tiles."""
    return lcm(page, 128)


class ClearBoundary:
    def __init__(self, tokens):
        self.tokens = tokens

    @cute.jit
    def __call__(
        self,
        k: cute.Tensor,
        v: cute.Tensor,
        table: cute.Tensor,
        limits: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(k, v, table, limits).launch(
            grid=(table.shape[0], cute.ceil_div(self.tokens * 512, 128), 1),
            block=(128, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self, k: cute.Tensor, v: cute.Tensor, table: cute.Tensor, limits: cute.Tensor
    ):
        request, block, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        value = block * 128 + tid
        token = value // 512
        if token < limits[request]:
            # vLLM's shared cache has large page strides. Widen before CuTe
            # multiplies the physical page by its stride (often beyond 2 GiB).
            page = Int64(table[request, token // k.shape[1]])
            offset = token % k.shape[1]
            head = (value // 128) % 4
            dim = value % 128
            k[page, offset, head, dim] = k.element_type(0)
            v[page, offset, head, dim] = v.element_type(0)


_COMPILED = Executables("replay_cache")


def clear_boundary(k, v, table, limits):
    # One executable serves every request count, table width and cache size.
    _COMPILED.launch(
        ClearBoundary(replay_cache_alignment(k.shape[1])),
        (k, v, table, limits),
        key=(replay_cache_alignment(k.shape[1]),),
        aligns=[4] * 4,
        dynamic=(2, 3),
    )
