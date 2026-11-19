# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""Served compressed routing never reads scratch it did not write.

Routing arenas and score scratch are `torch.empty` and come back from the CUDA
caching allocator holding earlier tensors' bytes. Each case fills the
allocator with 0xFF and then random bytes before building fresh served arenas
(as `Block8Impl` does) and runs source and reindex layers on rows on
both sides of every pool size. Results must match the reference (exact pool
and token selection up to ties, FP32 attention over the dequantized K/V) and
be bit-identical across fills.

Regression: the block-mean scans scored only rows past 16384 tokens, so with
an 8k pool, rows with 8k < context <= 16k selected their pool from stale
scores.
"""

import itertools

import pytest
import torch

from test_fp4_cache import dequant, rounded_nvfp4
from test_nvfp4_attention import _reference as attention_reference
from test_summary_policies import reference_pack

pytestmark = pytest.mark.gpu

PAGE = 128
LOCAL = 128
SINKS = 8
TOPK = 2048
SCALE = 128**-0.5
CELLS = [(16, 8192), (16, 16384), (16, 32768)]
# Prefill (prefix, queries): a single query, then requests whose rows cross
# the 8k, 16k and 32k pool sizes.
PREFILL = [(8, 1), (8192 - 150, 300), (16384 - 150, 300), (32768 - 150, 300)]
# Prefill whose contexts all fit in 16384 tokens: the 16k and 32k pools skip
# the block-mean scan, so no block score is written.
PREFILL_SHORT = [(8, 1), (8192 - 150, 300), (16384 - 300, 300)]
DECODE = [129, 8000, 8193, 12000, 16384, 16385, 24000, 32768, 32769, 40000]
_owners = itertools.count()


def poison(fill):
    """Leave large and small cached allocator blocks holding `fill` bytes."""
    torch.cuda.synchronize()
    blocks = [torch.empty(8 << 30, device="cuda", dtype=torch.uint8)]
    blocks += [
        torch.empty(256 << 10, device="cuda", dtype=torch.uint8) for _ in range(256)
    ]
    for block in blocks:
        if fill is None:
            block.random_(0, 256)
        else:
            block.fill_(fill)
    torch.cuda.synchronize()
    del blocks


def make_cache(compression, lengths, seed=29):
    from aleph_alpha_inference.kernels.summary_cache import SummaryCache
    from aleph_alpha_inference.kernels.vllm_backend import record_bytes, record_fields

    cache_type = SummaryCache
    assert compression == cache_type.BLOCK
    g = torch.Generator(device="cuda").manual_seed(seed)
    pages = [(n + PAGE - 1) // PAGE for n in lengths]
    # One record per page with 512-byte aligned fields, as the backend plans it.
    record = torch.zeros(
        sum(pages) + 3,
        record_bytes(cache_type, PAGE),
        device="cuda",
        dtype=torch.uint8,
    )
    cache = cache_type(
        *(
            record[:, start : start + width]
            for start, width in record_fields(cache_type, PAGE)
        ),
        page=PAGE,
    )
    perm = torch.randperm(sum(pages) + 3, device="cuda", generator=g).int()
    table = torch.zeros(len(lengths), max(pages) + 2, device="cuda", dtype=torch.int32)
    keys, offset = [], 0
    for row, (n, count) in enumerate(zip(lengths, pages)):
        table[row, :count] = perm[offset : offset + count]
        offset += count
        ids = torch.arange(n, device="cuda")
        slots = table[row, ids // PAGE].long() * PAGE + ids % PAGE
        k = torch.randn(n, 4, 128, device="cuda", dtype=torch.bfloat16, generator=g)
        v = torch.randn(n, 4, 128, device="cuda", dtype=torch.bfloat16, generator=g)
        for s in range(0, n, 12345):
            e = min(n, s + 12345)
            cache.append(k[s:e], v[s:e], slots[s:e])
        keys.append(k[:, 0])
    return cache, table, keys


def block_means(keys, compression):
    """Dequantized NVFP4 means of whole `compression`-token blocks."""
    count = keys.shape[0] // compression
    means = keys[: count * compression].float().view(count, compression, 128).mean(1)
    packed, scales = reference_pack(means)
    return dequant(packed, scales.view(torch.float8_e4m3fn).view(count, 8)).float()


def cut_slack(scores, chosen):
    """How far the chosen set is from a top set of `scores` (0 if exact)."""
    if not chosen.any() or chosen.all():
        return 0.0
    return max(0.0, float(scores[~chosen].max() - scores[chosen].min()))


def check_row(
    label, query, pool_query, keys, means, width, route, small, pool_tokens, compression
):
    """Route of one row of width `width` against the reference selection."""
    ids = route.long()
    assert ids.unique().numel() == ids.numel(), (label, "duplicate route ids")
    assert bool(((ids >= 0) & (ids < width)).all()), (label, "route out of bounds")
    total = TOPK + LOCAL + SINKS
    if width <= total:
        assert ids.numel() == width, (label, "short row attends everything")
        return
    assert ids.numel() == total, (label, "route length")
    fixed = torch.cat(
        [
            torch.arange(SINKS, device=ids.device),
            torch.arange(width - LOCAL, width, device=ids.device),
        ]
    )
    assert bool(torch.isin(fixed, ids).all()), (label, "sinks/local missing")
    remote = ids[~torch.isin(ids, fixed)]
    if width <= pool_tokens:
        pool = torch.arange(width, device=ids.device)
    else:
        # The pool: the top pool_tokens / B whole summary blocks by the source
        # query's block-mean scores (a trailing partial block scores 0).
        blocks = small[small >= 0].long()
        assert blocks.numel() == pool_tokens // compression, (label, "pool size")
        whole = width // compression
        scores = (pool_query @ means[:whole].T).relu().mean(0) * SCALE
        chosen = torch.zeros(whole, dtype=torch.bool, device=ids.device)
        chosen[blocks[blocks < whole]] = True
        tol = 1e-5 * (1 + float(scores.abs().max()))
        assert cut_slack(scores, chosen) <= tol, (
            label,
            "pool selection differs from reference",
        )
        pool = (
            blocks[:, None] * compression + torch.arange(compression, device=ids.device)
        ).flatten()
        pool = pool[pool < width]
    candidates = pool[(pool >= SINKS) & (pool < width - LOCAL)]
    assert bool(torch.isin(remote, candidates).all()), (
        label,
        "remote outside the pool",
    )
    assert remote.numel() == min(TOPK, candidates.numel()), (label, "remote count")
    scores = (query @ keys[candidates].T).relu().mean(0) * SCALE
    tol = 1e-5 * (1 + float(scores.abs().max()))
    assert cut_slack(scores, torch.isin(candidates, remote)) <= tol, (
        label,
        "token selection differs from reference",
    )


def run_layers(cache, table, q, cu, ends, starts, compression, pool_tokens, mode, span):
    """Source then reindex on fresh served arenas, as `Block8Impl` builds
    them; returns (role, query, out, route, used, small) per role."""
    from aleph_alpha_inference.kernels.serving_compressed import (
        routing_workspace,
        ServingCompressedLayer,
        source_pool,
    )
    from aleph_alpha_inference.kernels.sparse_decode import workspace_shapes

    decode = mode == "decode"
    rows = q.shape[0]
    owner = ("stale-scratch", next(_owners))
    scratch = routing_workspace(q.device, owner, rows, span, TOPK, decode, pool_tokens)
    common = dict(allow_staging=True, scratch=scratch, pool_tokens=pool_tokens)
    extra = dict(query_starts=())
    if decode:
        extra["attention_workspace"] = tuple(
            torch.empty(s, device="cuda", dtype=torch.float32)
            for s in workspace_shapes(rows, (TOPK + 136 + 63) // 64)
        )
    else:
        extra.update(query_starts=tuple(starts))
    results, pool = [], None
    g = torch.Generator(device="cuda").manual_seed(3)
    for role in ("source", "reindex"):
        layer = ServingCompressedLayer(
            cache,
            table,
            rows,
            span,
            TOPK,
            decode,
            source_pool=source_pool(
                q.device, owner, ("source", decode), rows, pool_tokens
            )
            if role == "source"
            else None,
            **common,
        )
        qr = (
            q
            if role == "source"
            else torch.randn(q.shape, device="cuda", dtype=q.dtype, generator=g)
        )
        out = torch.full_like(qr, float("nan"))
        layer.run(qr, cu, ends, role=role, pool=pool, out=out, bounds=[], **extra)
        torch.cuda.synchronize()
        if role == "source":
            pool = layer.pool
        results.append(
            (
                role,
                qr,
                out.clone(),
                layer.route.clone(),
                layer.used.clone(),
                layer.small.clone(),
            )
        )
    return results


@pytest.fixture
def fresh_arenas():
    from aleph_alpha_inference.kernels import serving_compressed

    yield
    # Drop this test's arenas (keyed by its private owners).
    for cached in (
        serving_compressed.routing_workspace,
        serving_compressed.source_pool,
    ):
        cached.cache_clear()
    torch.cuda.empty_cache()


@pytest.mark.parametrize("mode", ["prefill", "prefill-short", "decode"])
@pytest.mark.parametrize(
    "compression,pool_tokens", CELLS, ids=[f"c{c}-p{p // 1024}k" for c, p in CELLS]
)
def test_served_routing_ignores_stale_scratch(  # noqa: C901
    compression, pool_tokens, mode, fresh_arenas
):
    from aleph_alpha_inference.kernels.prefill_staging import stage_context

    torch.manual_seed(41)
    segments = {
        "decode": [(n - 1, 1) for n in DECODE],
        "prefill": PREFILL,
        "prefill-short": PREFILL_SHORT,
    }[mode]
    lengths = [p + n for p, n in segments]
    starts = [0]
    for _, n in segments:
        starts.append(starts[-1] + n)
    cache, table, keys = make_cache(compression, lengths)
    rows = starts[-1]
    q = torch.randn(rows, 1, 48, 128, device="cuda", dtype=torch.bfloat16)
    cu = torch.tensor(starts, device="cuda", dtype=torch.int32)
    ends = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    # The served span: the decode bound, or the prefill power-of-two bucket.
    span = (
        max(lengths)
        if mode == "decode"
        else max(16384, 1 << (max(lengths) - 1).bit_length())
    )

    request = torch.repeat_interleave(
        torch.arange(len(segments), device="cuda"), cu.diff()
    )
    width = ends[request] - cu[1:][request] + torch.arange(rows, device="cuda") + 1
    runs = []
    for fill in (0xFF, None):
        poison(fill)
        runs.append(
            run_layers(
                cache, table, q, cu, ends, starts, compression, pool_tokens, mode, span
            )
        )
    # Bit-identical across fills. Only source rows past the pool publish
    # summary blocks; the others leave `small` alone.
    for first, second in zip(*runs):
        role = first[0]
        for name, a, b in zip(("out", "route", "used", "small"), first[2:], second[2:]):
            if name == "small":
                if role != "source":
                    continue
                a, b = a[width > pool_tokens], b[width > pool_tokens]
            elif name == "route":
                live = torch.arange(a.shape[1], device="cuda") < first[4][:, None]
                a, b = a[live], b[live]
            assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), (
                mode,
                role,
                name,
                "depends on stale allocator bytes",
            )

    source_small = runs[0][0][5]
    sample = torch.Generator().manual_seed(0)
    for role, qr, out, route, used, small in runs[0]:
        for request, (start, stop) in enumerate(zip(starts, starts[1:])):
            count = stop - start
            picks = {0, count - 1} | set(
                torch.randint(0, count, (6,), generator=sample).tolist()
            )
            if count > 150:
                picks |= {149, 150, 151}  # The rows around the pool boundary.
            picks = sorted(p for p in picks if p < count)
            kr = rounded_nvfp4(keys[request])
            means = block_means(keys[request], compression)
            for i in picks:
                row = start + i
                width = lengths[request] - count + i + 1
                check_row(
                    f"{mode} {role} request {request} row {i}",
                    rounded_nvfp4(qr[row, 0, :12]),
                    rounded_nvfp4(q[row, 0, :12]),
                    kr[:width],
                    means,
                    width,
                    route[row, : int(used[row])],
                    source_small[row],
                    pool_tokens,
                    compression,
                )
            sel = torch.tensor([start + i for i in picks], device="cuda")
            kv = [
                x.clone()
                for x in stage_context(cache, table, ends, request, max(lengths))
            ]
            exact, bf16 = attention_reference(qr[sel], kv, route[sel], used[sel])
            got = out[sel].float()
            err, ref_err = (got - exact).abs(), (bf16.float() - exact).abs()
            assert err.max() <= 1.25 * ref_err.max() + 2**-8 * exact.abs().max(), (
                mode,
                role,
                request,
            )
