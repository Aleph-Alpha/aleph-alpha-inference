# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""NVFP4 fragment-dequant decode at the compressed serving shape, every split path."""

import math

import pytest
import torch


def _routes(lengths, table, route, used, page, num_pages):
    table.copy_(
        torch.randperm(num_pages, device="cuda")[: table.numel()].view(table.shape)
    )
    route.zero_()
    for row, length in enumerate(lengths):
        count = min(length, route.shape[1])
        used[row] = count
        if length <= route.shape[1]:
            ids = torch.arange(length, device="cuda")
        else:
            ids = torch.cat(
                (
                    torch.randperm(length - 136, device="cuda")[:2048] + 8,
                    torch.arange(length - 128, length, device="cuda"),
                    torch.arange(8, device="cuda"),
                )
            )
        route[row, :count] = ids.int()


@pytest.mark.gpu
@pytest.mark.parametrize("splits", [1, 2, 5, 35])
def test_fp4_split_decode_matches_reference(monkeypatch, splits):
    from aleph_alpha_inference.kernels import sparse_decode
    from aleph_alpha_inference.kernels.summary_cache import SummaryCache

    from test_decode_attention import _check

    torch.manual_seed(51)
    page, total = 128, 2184
    lengths = [0, 1, 37, 2183, 2184, 9000, 40000, 65536]
    batch = len(lengths)
    width = math.ceil(max(lengths) / page)
    num_pages = batch * width
    cache = SummaryCache.allocate(num_pages, page)
    key = torch.randn(num_pages * page, 4, 128, device="cuda", dtype=torch.bfloat16)
    cache.append(
        key, torch.randn_like(key), torch.arange(num_pages * page, device="cuda")
    )
    table = torch.empty(batch, width, device="cuda", dtype=torch.int32)
    route = torch.empty(batch, total, device="cuda", dtype=torch.int32)
    used = torch.empty(batch, device="cuda", dtype=torch.int32)
    requests = torch.arange(batch, device="cuda", dtype=torch.int32)
    _routes(lengths, table, route, used, page, num_pages)
    q = torch.randn(batch, 1, 48, 128, device="cuda", dtype=torch.bfloat16)

    monkeypatch.setattr(sparse_decode, "fp4_decode_splits", lambda *_: splits)
    attention = sparse_decode.SparseDecodeAttention(batch, cache, total=total)
    workspace = tuple(
        torch.empty(shape, device="cuda", dtype=torch.float32)
        for shape in sparse_decode.workspace_shapes(batch, attention.splits)
    )
    out = torch.empty_like(q)
    attention(q, table, route, used, requests, out=out, workspace=workspace)
    _check(q, cache, table, route, used, requests, out)

    # Graph replay with fresh inputs reuses the same executable and scratch.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        attention(q, table, route, used, requests, out=out, workspace=workspace)
    q.normal_()
    _routes(lengths[::-1], table, route, used, page, num_pages)
    graph.replay()
    _check(q, cache, table, route, used, requests, out)
