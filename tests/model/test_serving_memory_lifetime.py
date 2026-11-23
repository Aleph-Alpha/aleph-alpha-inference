# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""Serving caches must not retain completed requests."""

import gc

import pytest
import torch
from torch.multiprocessing.reductions import StorageWeakRef


@pytest.mark.gpu
def test_query_pack_does_not_retain_completed_query_storage():
    from aleph_alpha_inference.kernels.decode_pack import DecodeQueryPack

    pack = DecodeQueryPack()
    rows = 8
    packed = torch.empty(rows, 16, 64, device="cuda", dtype=torch.int8)
    scales = torch.empty(rows, 16, 2, device="cuda", dtype=torch.int32)
    torch.cuda.synchronize()
    q = torch.zeros(rows, 48, 128, device="cuda", dtype=torch.bfloat16)
    query_storage = StorageWeakRef(q.untyped_storage())
    pack(q, packed, scales)
    torch.cuda.synchronize()
    del q
    gc.collect()
    # Keep pack alive: a reusable kernel must not own an expired request's storage.
    assert query_storage.expired()


@pytest.mark.gpu
def test_sparse_decode_does_not_retain_completed_output_backing_storage():
    from aleph_alpha_inference.kernels.sparse_decode import (
        SparseDecodeAttention,
        workspace_shapes,
    )
    from aleph_alpha_inference.kernels.summary_cache import SummaryCache

    rows, page = 2, 128
    cache = SummaryCache(
        *(
            torch.zeros(1, width, device="cuda", dtype=torch.uint8)
            for width in SummaryCache.page_bytes(page)
        ),
        page=page,
    )
    attention = SparseDecodeAttention(rows, cache, total=644)
    workspace = tuple(
        torch.empty(shape, device="cuda", dtype=torch.float32)
        for shape in workspace_shapes(rows, attention.splits)
    )
    table = torch.zeros(rows, 1, device="cuda", dtype=torch.int32)
    routes = torch.zeros(rows, 644, device="cuda", dtype=torch.int32)
    used = torch.zeros(rows, device="cuda", dtype=torch.int32)
    requests = torch.arange(rows, device="cuda", dtype=torch.int32)
    q = torch.zeros(rows, 1, 48, 128, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    backing = torch.empty(256 * 1024, device="cuda", dtype=q.dtype)
    output_storage = StorageWeakRef(backing.untyped_storage())
    out = backing[: q.numel()].view_as(q)
    attention(q, table, routes, used, requests, out=out, workspace=workspace)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, torch.zeros_like(out), rtol=0, atol=0)
    del out, backing
    gc.collect()
    assert output_storage.expired()
