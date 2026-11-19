# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""Sparse decode attention over paged NVFP4 KV (kernel in `sparse_decode_fp4`)."""

from functools import lru_cache

import torch

from aleph_alpha_inference.kernels.cute_jit import Executables
from aleph_alpha_inference.kernels.sparse_decode_fp4 import (
    SparseDecodeFP4Split,
    SplitCombine,
)


# Fewest tokens of a split.
MIN_SPLIT_TOKENS = 64


def split_capacity(total):
    """Most splits a row of `total` routed tokens can use."""
    return (total + MIN_SPLIT_TOKENS - 1) // MIN_SPLIT_TOKENS


def workspace_shapes(batch, splits):
    """Shapes of the FP32 partial-output and reduction-statistic scratch for
    `batch` decode rows of up to `splits` splits each."""
    return ((batch * splits, 12, 4, 128), (batch * splits, 4, 12, 2))


# Executables are shared across decode batch sizes: eager DP decode sees every
# batch size, and per-size code grows host memory without bound.
_FP4_KERNELS = Executables("sparse_decode_fp4")
_FP4_COMBINES = Executables("sparse_decode_fp4_combine")


def fp4_decode_splits(batch, total, capacity):
    """Token splits per row, from B200 sweeps of 2184-token rows.

    Rows have uniform work: fill one wave of three CTAs per SM, no more
    (B=64: 6, B=128: 3, B=256: 1). Just past one wave, two splits halve the
    tail wave (B=512: 253 -> 217 us). Each split keeps at least
    `MIN_SPLIT_TOKENS` tokens; the combine handles up to 64 splits.
    """
    slots = 3 * _sm_count()
    splits = slots // batch if batch <= slots else 2 if batch < 2 * slots else 1
    return max(1, min(splits, capacity, total // MIN_SPLIT_TOKENS, 64))


@lru_cache(None)
def _sm_count():
    return torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count


class SparseDecodeAttention:
    """Attend each decode row over its `total` routed tokens of `cache`."""

    def __init__(self, batch, cache, *, total):
        self.total = total
        self.splits = split_capacity(total)
        if batch <= 0:
            raise ValueError("Sparse decode requires a positive batch size")
        self.batch = batch
        self.cache = cache
        self.kernel = SparseDecodeFP4Split(total)
        self.combine = SplitCombine()

    def _check_inputs(self, q, table, route, used, requests):
        if q.shape != (self.batch, 1, 48, 128) or q.dtype != torch.bfloat16:
            raise ValueError("Sparse decode requires BF16 queries [B, 1, 48, 128]")
        if q.device.type != "cuda" or q.device != self.cache.k.device:
            raise ValueError("Queries and cache must be on the same CUDA device")
        if (
            q.stride(-1) != 1
            or q.stride(-2) != 128
            or q.stride(0) % 8
            or q.data_ptr() % 16
        ):
            raise ValueError("Sparse decode requires aligned contiguous query heads")
        if (
            route.shape != (self.batch, self.total)
            or used.shape != (self.batch,)
            or requests.shape != (self.batch,)
        ):
            raise ValueError(
                "Sparse decode requires full routes and per-query metadata"
            )
        for tensor in (table, route, used, requests):
            if tensor.dtype != torch.int32 or tensor.device != q.device:
                raise ValueError(
                    "Sparse decode metadata must be INT32 on the query device"
                )
        if table.ndim != 2:
            raise ValueError("Sparse decode requires a two-dimensional page table")

    def _check_output(self, q, out):
        if out.shape != q.shape or out.dtype != q.dtype or out.device != q.device:
            raise ValueError(
                "Sparse decode output must match the query shape, dtype and device"
            )
        # A single split stores 16 bytes at a time.
        if (
            out.stride(-1) != 1
            or out.stride(-2) != 128
            or out.stride(0) % 8
            or out.data_ptr() % 16
        ):
            raise ValueError("Sparse decode requires aligned contiguous output heads")

    def _get_workspace(self, q, workspace):
        """The rows of `workspace` this batch uses, after validating it."""
        if len(workspace) != 2:
            raise ValueError(
                "Sparse decode requires partial-output and reduction-statistic scratch"
            )
        partial, stats = workspace
        partial_shape, stats_shape = workspace_shapes(self.batch, self.splits)
        needed = partial_shape[0]
        if (
            partial.shape[0] < needed
            or partial.shape[1:] != partial_shape[1:]
            or stats.shape[0] < needed
            or stats.shape[1:] != stats_shape[1:]
        ):
            raise ValueError(
                "Sparse decode scratch has insufficient capacity or an invalid shape"
            )
        for tensor in workspace:
            if (
                tensor.dtype != torch.float32
                or tensor.device != q.device
                or not tensor.is_contiguous()
                or tensor.data_ptr() % 16
            ):
                raise ValueError(
                    "Sparse decode scratch must be aligned contiguous FP32 on the query device"
                )
        partial, stats = partial[:needed], stats[:needed]
        return partial, stats

    def __call__(self, q, table, route, used, requests, *, out, workspace):
        """Write the attention output of every row to `out`.

        Args:
            q: Queries, [batch, 1, 48, 128] BF16.
            table: Pages of each request, [requests, pages] int32.
            route: Logical token positions each row attends, [batch, total]
                int32.
            used: Number of valid entries of each `route` row, int32. Rows
                without entries get zeros.
            requests: Request of each row, int32.
            out: Receives the output, shaped like `q`.
            workspace: FP32 scratch of the `workspace_shapes` of at least
                `batch` rows.

        Returns:
            `out`.

        Raises:
            ValueError: If a tensor has the wrong shape, dtype, device or
                alignment.
        """
        self._check_inputs(q, table, route, used, requests)
        self._check_output(q, out)
        partial, stats = self._get_workspace(q, workspace)
        out_view = out.view(self.batch, 48, 128)
        tensors = [
            q.view(self.batch, 48, 128),
            self.cache.k.view(torch.uint8),
            self.cache.v.view(torch.uint8),
            self.cache.ks,
            self.cache.vs,
            table,
            route,
            used,
            requests,
            partial,
            stats,
            out_view,
        ]
        # Splits are a run-time scalar: one executable serves every batch size.
        splits = fp4_decode_splits(self.batch, self.total, self.splits)
        _FP4_KERNELS.launch(
            self.kernel,
            tensors,
            key=(self.total,),
            scalars=(splits,),
            aligns=[4 if t.dtype == torch.int32 else 16 for t in tensors],
        )
        if splits > 1:
            _FP4_COMBINES.launch(
                self.combine,
                (partial, stats, out_view),
                scalars=(splits,),
                aligns=[16, 16, 8],
            )
        return out
