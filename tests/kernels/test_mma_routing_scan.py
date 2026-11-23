# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""The mma.sync routing scans reproduce the NVFP4 routing of the tcgen05 scans."""

import math

import pytest
import torch

from aleph_alpha_inference.kernels.arch import is_blackwell
from aleph_alpha_inference.kernels.decode_pack import DecodeQueryPack
from aleph_alpha_inference.kernels.routing_scan_mma import (
    KEYS_PER_CTA,
    MmaPoolScan,
    MmaSummaryScan,
)
from aleph_alpha_inference.kernels.serving_compressed import _COMPILED
from aleph_alpha_inference.kernels.summary_cache import SummaryCache

from serving_arena import new_owner, served_layer
from test_fp4_cache import dequant

pytestmark = pytest.mark.gpu

PAGE = 128
POOL = 16384


launch = _COMPILED.launch


def _cache(lengths, seed=7):
    """Random keys spread over the E4M3 scale range; each request's pages are
    a random slice of one allocation."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    pages = [math.ceil(n / PAGE) for n in lengths]
    cache = SummaryCache.allocate(sum(pages) + 2, PAGE)
    perm = torch.randperm(sum(pages) + 2, device="cuda", generator=g).int()
    table = torch.zeros(len(lengths), max(pages) + 1, device="cuda", dtype=torch.int32)
    offset = 0
    for row, (n, count) in enumerate(zip(lengths, pages)):
        table[row, :count] = perm[offset : offset + count]
        offset += count
        ids = torch.arange(n, device="cuda")
        slots = table[row, ids // PAGE].long() * PAGE + ids % PAGE
        k = torch.randn(n, 4, 128, device="cuda", dtype=torch.bfloat16, generator=g)
        k = (
            k
            * torch.logspace(-1, 1, n, device="cuda", dtype=torch.bfloat16)[
                :, None, None
            ]
        )
        cache.append(k, torch.randn_like(k), slots)
    return cache, table


def _pack_queries(q):
    rows = q.shape[0]
    qp = torch.zeros(rows, 16, 64, device="cuda", dtype=torch.int8)
    qs = torch.zeros(rows, 16, 2, device="cuda", dtype=torch.int32)
    DecodeQueryPack()(q, qp, qs)
    return qp, qs


def _dequant_queries(qp, qs):
    """[rows, 12, 128] FP32 values of the packed routing heads."""
    return dequant(
        qp[:, :12].view(torch.uint8), qs[:, :12].view(torch.float8_e4m3fn)
    ).float()


def _dequant_keys(k, ks, pages, slots):
    return dequant(
        k[pages, slots].view(torch.uint8), ks[pages, slots].view(torch.float8_e4m3fn)
    ).float()


def _scores(qd, kd, live):
    s = torch.relu(qd @ kd.T).sum(0) * (128**-0.5 / 12)
    return torch.where(live, s, torch.full_like(s, -float("inf")))


def _pool_reference(qd, cache, table, request, pool_row, hi):
    blocks = pool_row.long()
    tokens = blocks[:, None] * 8 + torch.arange(8, device="cuda")
    live = ((blocks[:, None] >= 0) & (tokens < hi)).flatten()
    tokens = tokens.clamp(min=0).flatten()
    pages = table[request, tokens // PAGE].long()
    return _scores(qd, _dequant_keys(cache.rk, cache.rks, pages, tokens % PAGE), live)


def _summary_reference(qd, cache, table, request, width, hi):
    s = torch.arange(width, device="cuda")
    # Only the summaries the row's context completes, as the scans score.
    live = (s + 1) * cache.BLOCK <= hi
    per_page = PAGE // cache.BLOCK
    pages = table[request, (s // per_page).clamp(max=table.shape[1] - 1)].long()
    keys = _dequant_keys(cache.summaries, cache.summary_scales, pages, s % per_page)
    return _scores(qd, keys, live)


def _rows(lengths, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    requests = torch.tensor([0, 1, 2, 3, 3, 1], device="cuda", dtype=torch.int32)
    hi = torch.tensor(lengths, device="cuda")[requests.long()].int()
    rows = requests.numel()
    q = torch.randn(rows, 48, 128, device="cuda", dtype=torch.bfloat16, generator=g)
    pool = torch.full((rows, POOL // 8), -1, device="cuda", dtype=torch.int32)
    for r in range(rows):
        blocks = math.ceil(int(hi[r]) / 8)
        if blocks > POOL // 8:
            pool[r] = torch.randperm(blocks, device="cuda", generator=g)[
                : POOL // 8
            ].int()
        else:
            pool[r, :blocks] = torch.arange(blocks, device="cuda").int()
    return requests, hi, q * 3, pool


def _views(cache):
    return (
        cache.rk.view(torch.uint8),
        cache.rks,
        cache.summaries,
        cache.summary_scales,
    )


def test_mma_scans_match_exact_nvfp4_scores():
    lengths = [100, 3000, 16384, 70000]
    cache, table = _cache(lengths)
    requests, hi, q, pool = _rows(lengths)
    qp, qs = _pack_queries(q)
    qd = _dequant_queries(qp, qs)
    rk, rks, summaries, summary_scales = _views(cache)
    rows = requests.numel()

    scores = torch.full((rows, POOL), float("nan"), device="cuda")
    launch(
        MmaPoolScan(),
        (qp, qs, rk, rks, table, requests, pool, hi, scores),
        key=("pool",),
    )
    for r in range(rows):
        ref = _pool_reference(
            qd[r], cache, table, int(requests[r]), pool[r], int(hi[r])
        )
        # Same products as the tcgen05 scans; FP32 sums in another order.
        torch.testing.assert_close(scores[r], ref, rtol=1e-4, atol=1e-5)

    width = math.ceil(max(lengths) / cache.BLOCK / KEYS_PER_CTA) * KEYS_PER_CTA
    scores = torch.full((rows, width), float("nan"), device="cuda")
    launch(
        MmaSummaryScan(POOL),
        (qp, qs, summaries, summary_scales, table, requests, pool, hi, scores),
        key=("summary",),
    )
    for r in range(rows):
        if int(hi[r]) <= POOL:
            assert scores[r].isnan().all(), "rows that fit the pool are not scored"
            continue
        ref = _summary_reference(
            qd[r], cache, table, int(requests[r]), width, int(hi[r])
        )
        torch.testing.assert_close(scores[r], ref, rtol=1e-4, atol=1e-5)


def test_mma_scans_match_tcgen05_scans():
    if not is_blackwell():
        pytest.skip("compares against the tcgen05 scans")
    from aleph_alpha_inference.kernels.decode_reindex import SelectedPoolScan
    from aleph_alpha_inference.kernels.summary_scan import BlockMeanScan

    lengths = [100, 3000, 16384, 70000]
    cache, table = _cache(lengths)
    requests, hi, q, pool = _rows(lengths)
    qp, qs = _pack_queries(q)
    rk, rks, summaries, summary_scales = _views(cache)
    rows = requests.numel()

    tensors = (qp, qs, rk, rks, table, requests, pool, hi)
    expected = torch.empty(rows, POOL, device="cuda")
    launch(SelectedPoolScan(pages=8), (*tensors, expected), key=(0, 8))
    actual = torch.empty_like(expected)
    launch(MmaPoolScan(), (*tensors, actual), key=("pool",))
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)

    width = math.ceil(max(lengths) / 16 / 128) * 128
    tensors = (qp, qs, summaries, summary_scales, table, requests, pool, hi)
    expected = torch.full((rows, width), float("nan"), device="cuda")
    launch(BlockMeanScan(POOL), (*tensors, expected), key=(POOL,))
    actual = torch.full_like(expected, float("nan"))
    launch(MmaSummaryScan(POOL), (*tensors, actual), key=("summary",))
    # Past a row's complete summaries, BlockMeanScan publishes 0 and
    # MmaSummaryScan -inf. Selection never reaches them: a row scored here has
    # more than POOL tokens, so at least the POOL // 16 summaries it keeps.
    live = torch.arange(width, device="cuda") < (hi // 16)[:, None]
    scored = (hi > POOL)[:, None] & live
    torch.testing.assert_close(actual[scored], expected[scored], rtol=1e-4, atol=1e-5)


def test_query_pack_matches_cache_quantization():
    """Queries and keys round to NVFP4 identically on every architecture:
    E4M3 scales of amax / 6 and round-to-nearest-even E2M1 values, including
    the ties at quarter steps."""
    g = torch.Generator(device="cuda").manual_seed(3)
    rows = 64
    q = torch.randn(rows, 48, 128, device="cuda", dtype=torch.bfloat16, generator=g)
    # Rows 0-15: ties at every quarter step with unit scales (amax 6).
    ties = torch.randint(-24, 25, (16, 48, 128), device="cuda", generator=g) * 0.25
    ties[..., 0] = 6
    q[:16] = ties.bfloat16()
    # Rows 16-31: the E4M3 scale range, down to the clamped minimum.
    q[16:32] *= torch.logspace(-4, 3, 16, device="cuda")[:, None, None].bfloat16()
    qp, qs = _pack_queries(q)

    cache = SummaryCache.allocate(math.ceil(rows * 12 / PAGE))
    keys = q[:, :12].reshape(rows * 12, 1, 128).expand(-1, 4, -1).contiguous()
    slots = torch.arange(rows * 12, device="cuda")
    cache.append(keys, keys, slots)
    pages, offsets = slots // PAGE, slots % PAGE
    packed = cache.rk[pages, offsets].view(torch.uint8).view(rows, 12, 64)
    scales = cache.rks[pages, offsets].view(rows, 12, 2)
    torch.testing.assert_close(qp[:, :12].view(torch.uint8), packed, rtol=0, atol=0)
    torch.testing.assert_close(qs[:, :12], scales, rtol=0, atol=0)


def _assert_top(selected, scores, count):
    """`selected` holds `count` ids whose scores are the largest in `scores`."""
    assert selected.numel() == count
    chosen = torch.zeros_like(scores, dtype=torch.bool)
    chosen[selected.long()] = True
    assert chosen.sum() == count, "duplicate ids"
    if count < scores.numel():
        assert scores[chosen].min() >= scores[~chosen].max()


@pytest.mark.parametrize("decode", [True, False])
def test_compressed_layer_routes_from_exact_scores(decode):
    """`ServingCompressedLayer.select` picks the pool and the routes that the
    exact NVFP4 scores order first, for decode rows and prefill rows alike."""
    lengths = [100, 3000, 20000, 70000]
    cache, table = _cache(lengths)
    g = torch.Generator(device="cuda").manual_seed(5)
    if decode:
        counts = [1, 1, 1, 1]
    else:
        counts = [3, 40, 61, 70]
    starts = [0]
    for count in counts:
        starts.append(starts[-1] + count)
    rows = starts[-1]
    q = torch.randn(rows, 1, 48, 128, device="cuda", dtype=torch.bfloat16, generator=g)
    cu = torch.tensor(starts, device="cuda", dtype=torch.int32)
    ends = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    requests = torch.repeat_interleave(
        torch.arange(4, device="cuda"), torch.tensor(counts, device="cuda")
    )
    # Row i of a request attends its first `ends - count + i + 1` tokens.
    hi = ends[requests] - torch.tensor(counts, device="cuda")[requests]
    hi = hi + torch.arange(rows, device="cuda") - cu[requests] + 1
    topk, local, sinks = 2048, 128, 8
    span = 1 << (max(lengths) - 1).bit_length()
    layer = served_layer(
        cache, table, rows, span, topk, decode, owner=new_owner(), source=True
    )
    layer.select(q, cu, ends, role="source", bounds=[])
    qd = _dequant_queries(layer.qp, layer.qs).view(rows, 12, 128)
    for r in range(rows):
        request, length = int(requests[r]), int(hi[r])
        assert int(layer.hi[r]) == length
        if length > POOL:
            width = math.ceil(length / 16)
            summary = _summary_reference(qd[r], cache, table, request, width, length)
            _assert_top(layer.small[r], summary, POOL // 16)
        else:
            # An identity pool of whole summaries; the scan masks their tail.
            blocks = math.ceil(length / cache.BLOCK) * (cache.BLOCK // 8)
            assert torch.equal(
                layer.pool[r, :blocks], torch.arange(blocks, device="cuda").int()
            )
            assert bool((layer.pool[r, blocks:] == -1).all())
        scores = _pool_reference(qd[r], cache, table, request, layer.pool[r], length)
        tokens = (
            layer.pool[r].long().repeat_interleave(8) * 8
            + torch.arange(POOL, device="cuda") % 8
        )
        used = int(layer.used[r])
        route = layer.route[r, :used]
        if length <= topk + local + sinks:
            assert used == length
            assert torch.equal(
                route.sort().values, torch.arange(length, device="cuda").int()
            )
            continue
        assert used == topk + local + sinks
        assert torch.equal(
            route[topk : topk + local],
            torch.arange(length - local, length, device="cuda").int(),
        )
        assert torch.equal(
            route[topk + local :], torch.arange(sinks, device="cuda").int()
        )
        remote = torch.where(
            (tokens >= sinks) & (tokens < length - local), scores, -torch.inf
        )
        by_token = torch.full((length,), -torch.inf, device="cuda")
        live = remote > -torch.inf
        by_token[tokens[live]] = remote[live]
        _assert_top(route[:topk], by_token, topk)
