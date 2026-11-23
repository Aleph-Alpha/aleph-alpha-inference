# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""Grouped prefill scans reproduce the per-query indexer scans bit for bit."""

import pytest
import torch


def _fill(cache, generator):
    """Random NVFP4 routing keys and summaries with finite E4M3 scales."""

    def fill(region, scale_start):
        region[:, :scale_start] = torch.randint(
            0,
            256,
            region[:, :scale_start].shape,
            generator=generator,
            device="cuda",
            dtype=torch.uint8,
        )
        region[:, scale_start:] = torch.randint(
            0x28,
            0x40,
            region[:, scale_start:].shape,
            generator=generator,
            device="cuda",
            dtype=torch.uint8,
        )

    fill(cache.routing, 64 * 128)
    blocks = 128 // cache.BLOCK
    fill(cache.tensors[2][:, : blocks * 72], blocks * 64)


@pytest.mark.gpu
def test_grouped_scans_match_per_query_scans():  # noqa: PLR0915
    from aleph_alpha_inference.kernels.decode_pack import DecodeQueryPack
    from aleph_alpha_inference.kernels.decode_reindex import SelectedPoolScan
    from aleph_alpha_inference.kernels.decode_selection import (
        ExpandedPoolSelection,
        RadixSelection,
    )
    from aleph_alpha_inference.kernels.serving_compressed import _COMPILED, row_bounds
    from aleph_alpha_inference.kernels.summary_cache import SummaryCache
    from aleph_alpha_inference.kernels.summary_scan import (
        BlockMeanScan,
        GroupedBlockMeanScan,
        GroupedShortScan,
    )

    launch = _COMPILED.launch

    generator = torch.Generator(device="cuda").manual_seed(3)
    # Uneven query counts give groups that straddle requests; contexts cover
    # short (identity pool) and long rows, partial pages and partial blocks.
    queries = [5, 3, 301, 1, 200, 2]
    contexts = [40000, 9000, 16390, 70001, 1003, 16384]
    pages = [(n + 127) // 128 for n in contexts]
    cache = SummaryCache.allocate(sum(pages) + 3, 128)
    _fill(cache, generator)
    perm = torch.randperm(sum(pages) + 3, device="cuda", dtype=torch.int32)
    table = torch.zeros(len(contexts), max(pages), device="cuda", dtype=torch.int32)
    offset = 0
    for i, n in enumerate(pages):
        table[i, :n] = perm[offset : offset + n]
        offset += n
    rows = sum(queries)
    cu = torch.tensor(
        [0, *torch.tensor(queries).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )
    ends = torch.tensor(contexts, device="cuda", dtype=torch.int32)
    q = torch.randn(
        rows, 48, 128, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    qp = torch.zeros(rows, 16, 64, device="cuda", dtype=torch.int8)
    qs = torch.zeros(rows, 16, 2, device="cuda", dtype=torch.int32)
    requests = torch.zeros(rows, device="cuda", dtype=torch.int32)
    hi = torch.zeros_like(requests)
    DecodeQueryPack()(q, qp, qs)
    row_bounds(cu, ends, requests, hi)
    c = cache.BLOCK
    width = (131072 + c * 128 - 1) // (c * 128) * 128
    block_args = (
        qp.view(torch.uint8),
        qs,
        cache.summaries,
        cache.summary_scales,
        table,
        requests,
        None,
        hi,
    )
    small = torch.empty(rows, 16384 // c, device="cuda", dtype=torch.int32)
    block_args = block_args[:6] + (small, hi)
    expected = torch.full((rows, width), float("nan"), device="cuda")
    got = expected.clone()
    launch(BlockMeanScan(), (*block_args, expected), key=(c,))
    assert not expected[hi > 16384, :1].isnan().any()
    launch(GroupedBlockMeanScan(), (*block_args, got), key=(c,))
    assert torch.equal(expected.view(torch.int32), got.view(torch.int32))
    # Pool selection and expansion as served.
    launch(
        RadixSelection(0, threads=512, topk=16384 // c, block_size=c),
        (expected, small, hi, small, small, hi),
        key=(16384 // c, c, False),
    )
    # Every summary block expands to its c / 8 key blocks.
    factor = c // 8
    steps = torch.arange(factor, device="cuda", dtype=torch.int32)
    blocks = (small * factor)[:, :, None] + steps
    pool = torch.where(small[:, :, None] >= 0, blocks, -1).view(rows, 2048)
    # Fused pool selection and expansion.
    fused_small = torch.full_like(small, -7)
    fused_pool = torch.full_like(pool, -7)
    launch(
        ExpandedPoolSelection(16384 // c, c),
        (expected, fused_small, hi, fused_small, fused_pool, hi),
        key=(16384 // c, c, False),
    )
    assert torch.equal(fused_small, small)
    assert torch.equal(fused_pool, pool)
    pool_args = (
        qp.view(torch.uint8),
        qs,
        cache.rk.view(torch.uint8),
        cache.rks,
        table,
        requests,
        pool,
        hi,
    )
    expected = torch.full((rows, 16384), float("nan"), device="cuda")
    got = expected.clone()
    # The decode pool scan scores every row itself and is the reference.
    launch(SelectedPoolScan(pages=8), (*pool_args, expected), key=("reference",))
    assert not expected.isnan().any()
    launch(GroupedShortScan(16384), (*pool_args, got), key=(16384,))
    launch(
        SelectedPoolScan(short_tokens=16384, pages=16),
        (*pool_args, got),
        key=(16384, 16),
    )
    assert torch.equal(expected.view(torch.int32), got.view(torch.int32))


@pytest.mark.gpu
@pytest.mark.parametrize("distinct", [3, 1 << 20])
def test_segmented_pool_selection_matches_single_pass(distinct):
    """Segment top-k then merge equals one pass, ties broken by block index."""
    from aleph_alpha_inference.kernels.decode_selection import (
        ExpandedPoolSelection,
        RadixSelection,
    )
    from aleph_alpha_inference.kernels.serving_compressed import _COMPILED

    launch = _COMPILED.launch

    generator = torch.Generator(device="cuda").manual_seed(distinct)
    rows, width, segments, c, k = 9, 65536, 8, 16, 1024
    scores = (
        torch.randint(
            0, distinct, (rows, width), device="cuda", generator=generator
        ).float()
        / 7
    )
    hi = torch.tensor(
        [1 << 20, 1000000, 524289, 16384, 16385, 17000, 999999, 5, 131072 * 8 - 3],
        device="cuda",
        dtype=torch.int32,
    )
    lo = torch.zeros_like(hi)
    expected_small = torch.full((rows, k), -7, device="cuda", dtype=torch.int32)
    expected_pool = torch.full((rows, 2 * k), -7, device="cuda", dtype=torch.int32)
    launch(
        ExpandedPoolSelection(k, c),
        (scores, expected_small, hi, expected_small, expected_pool, lo),
        key=(k, c, True, False),
    )
    candidates = torch.full((rows, segments * k), -7, device="cuda", dtype=torch.int32)
    segment_out = candidates.view(rows * segments, k)
    launch(
        RadixSelection(0, threads=512, segments=segments, topk=k, block_size=c),
        (
            scores.view(rows * segments, width // segments),
            segment_out,
            hi,
            segment_out,
            segment_out,
            hi,
        ),
        key=(k, c, segments),
    )
    small = torch.full_like(expected_small, -7)
    pool = torch.full_like(expected_pool, -7)
    launch(
        ExpandedPoolSelection(k, c, merge=True),
        (scores, candidates, hi, small, pool, hi),
        key=(k, c, "merge"),
    )
    assert torch.equal(small, expected_small)
    assert torch.equal(pool, expected_pool)
