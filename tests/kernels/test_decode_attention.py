# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""Paged decode correctness, mutable graph inputs and pointer rebinding."""

import math

import pytest
import torch


def _routes(lengths, requests, table, route, used, num_pages, page):
    full = torch.randperm(num_pages, device=table.device, dtype=torch.int64)
    full = full[: table.numel()].view(table.shape).int()
    for request in range(table.shape[0]):
        largest = max(
            (
                lengths[row]
                for row, value in enumerate(requests.tolist())
                if value == request
            ),
            default=0,
        )
        full[request, math.ceil(largest / page) :] = -1
    table.copy_(full)
    route.zero_()
    used.copy_(
        torch.tensor(
            [min(length, 644) for length in lengths],
            device=used.device,
            dtype=torch.int32,
        )
    )
    for row, length in enumerate(lengths):
        if length <= 644:
            ids = torch.arange(length, device=route.device)
        else:
            ids = torch.cat(
                (
                    torch.randperm(length - 132, device=route.device)[:512] + 4,
                    torch.arange(length - 128, length, device=route.device),
                    torch.arange(4, device=route.device),
                )
            )
        route[row, : len(ids)].copy_(ids)


def _check(q, cache, table, route, used, requests, out):
    for row in range(q.shape[0]):
        count = int(used[row])
        if count == 0:
            torch.testing.assert_close(
                out[row], torch.zeros_like(out[row]), rtol=0, atol=0
            )
            continue
        logical = route[row, :count].long()
        pages = table[requests[row], logical // cache.page].long()
        slots = logical % cache.page
        from test_fp4_cache import dequant

        key = dequant(cache.k[pages, slots], cache.ks[pages, slots]).float()
        value = dequant(cache.v[pages, slots], cache.vs[pages, slots]).float()
        key = key.repeat_interleave(12, dim=1)
        value = value.repeat_interleave(12, dim=1)
        if count == 1:
            torch.testing.assert_close(out[row, 0], value[0].bfloat16(), rtol=0, atol=0)
            continue
        scores = torch.einsum("hd,khd->hk", q[row, 0].float(), key) * 128**-0.5
        probabilities = scores.softmax(-1)
        expected = torch.einsum("hk,khd->hd", probabilities, value)
        magnitude = torch.einsum("hk,khd->hd", probabilities, value.abs())
        # BF16 unnormalized probabilities and final BF16 output each round.
        # sum(p * abs(v)) accounts for cancellation; the last term covers FP32
        # reduction order. This is not a claim of FP32-oracle bit equivalence.
        bound = expected.abs() / 255 + 0.0042 * magnitude + 2e-5
        error = (out[row, 0].float() - expected).abs()
        assert bool((error <= bound).all()), float((error / bound).max())


def _mutate_attention_cache(cache):
    # Cache-policy hints must observe data and scale writes between replays.
    cache.k.view(torch.uint8).bitwise_xor_(128)
    cache.v.view(torch.uint8).bitwise_xor_(128)
    cache.ks.copy_((cache.ks.float() * 0.75).to(cache.ks.dtype))
    cache.vs.copy_((cache.vs.float() * 1.25).to(cache.vs.dtype))


@pytest.mark.gpu
def test_decode_rebinds_layouts_and_replays_mutable_graphs():
    from aleph_alpha_inference.kernels.sparse_decode import (
        SparseDecodeAttention,
        workspace_shapes,
    )
    from aleph_alpha_inference.kernels.summary_cache import SummaryCache as ServingCache

    torch.manual_seed(914)
    page, batch = 128, 8
    lengths = [0, 1, 7, 128, 644, 645, 8193, 17017]
    width = math.ceil(max(lengths) / page)
    num_pages = batch * width
    # Additional per-page padding exercises allocator layouts with a larger
    # physical stride than the packed payload.
    cache = ServingCache(
        *(
            torch.empty(num_pages, width + 512, device="cuda", dtype=torch.uint8)
            for width in ServingCache.page_bytes(page)
        ),
        page=page,
    )
    # Fill every packed E2M1 byte pattern with finite E4M3 scales.
    for values in (cache.k, cache.v):
        values.copy_(torch.randint_like(values, 256))
    for scales in (cache.ks, cache.vs):
        scales.view(torch.uint8).copy_(
            torch.randint(0x20, 0x30, scales.shape, device="cuda", dtype=torch.uint8)
        )
    attention = SparseDecodeAttention(batch, cache, total=644)
    reference_attention = SparseDecodeAttention(batch, cache, total=644)
    workspace = tuple(
        torch.empty(shape, device="cuda", dtype=torch.float32)
        for shape in workspace_shapes(batch, attention.splits)
    )
    reference_workspace = tuple(torch.empty_like(t) for t in workspace)
    captured = []
    for padded in (False, True):
        step = 2 if padded else 1
        q = torch.randn(batch * step, 1, 48, 128, device="cuda", dtype=torch.bfloat16)[
            ::step
        ]
        out = (
            torch.empty_like(q)
            if not padded
            else torch.empty(batch * 2, 1, 48, 128, device="cuda", dtype=q.dtype)[::2]
        )
        table = torch.empty(
            batch, width + (4 if padded else 0), device="cuda", dtype=torch.int32
        )[:, :width]
        route = torch.empty(
            batch, 644 + (8 if padded else 0), device="cuda", dtype=torch.int32
        )[:, :644]
        used = torch.empty(batch * step, device="cuda", dtype=torch.int32)[::step]
        live = torch.tensor(lengths, device="cuda", dtype=torch.int32)
        requests = torch.tensor(
            [0, 1, 2, 3, 4, 4, 6, 7], device="cuda", dtype=torch.int32
        )
        _routes(lengths, requests, table, route, used, num_pages, page)
        assert (
            attention(q, table, route, used, requests, out=out, workspace=workspace)
            is out
        )
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            attention(q, table, route, used, requests, out=out, workspace=workspace)
        captured.append((graph, q, table, route, used, live, requests, out))
    # Both graphs retain their own q/table/output pointers after the wrapper
    # has rebound to the other layout; the split scratch is shared serially.
    for trial in range(3):
        _mutate_attention_cache(cache)
        for graph, q, table, route, used, live, requests, out in captured:
            q.normal_()
            current_lengths = (
                lengths[:1] + lengths[1 + trial :] + lengths[1 : 1 + trial]
            )
            _routes(current_lengths, requests, table, route, used, num_pages, page)
            live.copy_(torch.tensor(current_lengths, device="cuda", dtype=torch.int32))
            graph.replay()
            _check(q, cache, table, route, used, requests, out)
            expected = reference_attention(
                q,
                table,
                route,
                used,
                requests,
                out=torch.empty(q.shape, device="cuda", dtype=q.dtype),
                workspace=reference_workspace,
            )
            torch.testing.assert_close(out, expected, rtol=0, atol=0)
    # Vector accesses require every row, not just the base pointer, to align.
    bad_rows = torch.empty(batch, 48 * 128 + 1, device="cuda", dtype=q.dtype)
    bad_rows = bad_rows[:, : 48 * 128].view(batch, 1, 48, 128)
    with pytest.raises(ValueError, match="aligned contiguous query heads"):
        attention(bad_rows, table, route, used, requests, out=out, workspace=workspace)
    with pytest.raises(ValueError, match="aligned contiguous output heads"):
        attention(q, table, route, used, requests, out=bad_rows, workspace=workspace)
