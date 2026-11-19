# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""Direct NVFP4 prefill attention matches BF16 attention over dequantized KV."""

import pytest
import torch

pytestmark = pytest.mark.gpu

PAGE = 128


def _cache(lengths, seed=11):
    from aleph_alpha_inference.kernels.summary_cache import SummaryCache
    from aleph_alpha_inference.kernels.vllm_backend import record_bytes, record_fields

    g = torch.Generator(device="cuda").manual_seed(seed)
    pages = [(n + PAGE - 1) // PAGE for n in lengths]
    # One record per page, so each field is a strided view as vLLM plans it.
    record = torch.zeros(
        sum(pages) + 3,
        record_bytes(SummaryCache, PAGE),
        device="cuda",
        dtype=torch.uint8,
    )
    cache = SummaryCache(
        *(
            record[:, start : start + width]
            for start, width in record_fields(SummaryCache, PAGE)
        ),
        page=PAGE,
    )
    perm = torch.randperm(sum(pages) + 3, device="cuda", generator=g).int()
    table = torch.zeros(len(lengths), max(pages) + 2, device="cuda", dtype=torch.int32)
    offset = 0
    for row, (n, count) in enumerate(zip(lengths, pages)):
        table[row, :count] = perm[offset : offset + count]
        offset += count
        ids = torch.arange(n, device="cuda")
        slots = table[row, ids // PAGE].long() * PAGE + ids % PAGE
        k = torch.randn(n, 4, 128, device="cuda", dtype=torch.bfloat16, generator=g)
        # Spread scales over the E4M3 range the cache stores.
        k = (
            k
            * torch.logspace(-2, 2, n, device="cuda", dtype=torch.bfloat16)[
                :, None, None
            ]
        )
        v = torch.randn(n, 4, 128, device="cuda", dtype=torch.bfloat16, generator=g)
        cache.append(k, 3 * v, slots)
    return cache, table


def _reference(q, kv, route, used):
    """FP32 attention and BF16 attention over dequantized K/V.

    The BF16 result follows a BF16 tensor-core kernel: FP32 scores, the
    unnormalized probabilities rounded to BF16, an FP32 output sum normalized
    by the FP32 probability sum, and the output rounded to BF16.
    """
    k, v = kv
    exact = torch.empty_like(q, dtype=torch.float32)
    bf16 = torch.empty_like(q)
    for r in range(q.shape[0]):
        idx = route[r, : int(used[r])].long()
        s = torch.einsum(
            "hjd,nhd->hjn", q[r, 0].float().view(4, 12, 128), k[idx, 0].float()
        )
        s = s * 128**-0.5
        values = v[idx, 0].float()
        exact[r, 0] = torch.einsum("hjn,nhd->hjd", s.softmax(-1), values).reshape(
            48, 128
        )
        unnormalized = (s - s.amax(-1, keepdim=True)).exp()
        rounded = torch.einsum(
            "hjn,nhd->hjd", unnormalized.bfloat16().float(), values
        ) / unnormalized.sum(-1, keepdim=True)
        bf16[r, 0] = rounded.reshape(48, 128).bfloat16()
    return exact, bf16


@pytest.mark.parametrize("q_scale", [1.0, 64.0, 1 / 64])
def test_matches_bf16_attention_over_dequantized_kv(q_scale):
    from aleph_alpha_inference.kernels.nvfp4_attention import nvfp4_attention
    from aleph_alpha_inference.kernels.prefill_staging import stage_context

    lengths = [37, 5000, 20001]
    cache, table = _cache(lengths)
    g = torch.Generator(device="cuda").manual_seed(3)
    width = 2184
    counts = [3, 40, 61]
    rows = sum(counts)
    requests = torch.repeat_interleave(
        torch.arange(3, device="cuda"), torch.tensor(counts, device="cuda")
    ).int()
    span = torch.tensor(lengths, device="cuda")[requests.long()]
    route = (torch.rand(rows, width, device="cuda", generator=g) * span[:, None]).int()
    # Ragged: one key, partial and full 32-key blocks, every key, and no key.
    used = torch.minimum(
        torch.randint(1, width + 1, (rows,), device="cuda", generator=g).int(),
        span.int(),
    )
    used[:4] = torch.tensor([1, 31, 33, 0], device="cuda", dtype=torch.int32)
    used[-1] = width
    q = q_scale * torch.randn(
        rows, 1, 48, 128, device="cuda", dtype=torch.bfloat16, generator=g
    )
    out = torch.full_like(q, float("nan"))
    nvfp4_attention(q, cache, table, requests, route, used, out=out)
    lengths_t = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    begin = 0
    for request, count in enumerate(counts):
        sl = slice(begin, begin + count)
        begin += count
        kv = [x.clone() for x in stage_context(cache, table, lengths_t, request, 32768)]
        live = used[sl] > 0
        exact, bf16 = _reference(q[sl][live], kv, route[sl][live], used[sl][live])
        got = out[sl][live].float()
        # As accurate as the BF16 kernel: both round P (FP16 vs BF16) and O.
        err, ref_err = (got - exact).abs(), (bf16.float() - exact).abs()
        assert err.max() <= 1.25 * ref_err.max() + 1e-6
        assert err.mean() <= 1.25 * ref_err.mean()
        torch.testing.assert_close(got, bf16.float(), atol=2e-2, rtol=2e-2)
        assert bool((out[sl][~live] == 0).all())


def test_compressed_prefill_mixes_union_tiles_and_rows():
    from aleph_alpha_inference.kernels.prefill_staging import stage_context

    from serving_arena import new_owner, served_layer

    torch.manual_seed(5)
    # Single-query requests and the tail of the last request use the per-row
    # kernel; the rest run as union tiles.
    lengths, counts = [9, 20001, 4097, 40000], [1, 300, 1, 200]
    cache, table = _cache(lengths)
    starts = [0]
    for count in counts:
        starts.append(starts[-1] + count)
    rows = starts[-1]
    q = torch.randn(rows, 1, 48, 128, device="cuda", dtype=torch.bfloat16)
    cu = torch.tensor(starts, device="cuda", dtype=torch.int32)
    ends = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    pool, owner = None, new_owner()
    for role in ("source", "reindex"):
        layer = served_layer(
            cache, table, rows, 65536, 2048, False, owner=owner, source=role == "source"
        )
        out = torch.full_like(q, float("nan"))
        layer.run(
            q, cu, ends, role=role, pool=pool, out=out, bounds=[], query_starts=starts
        )
        pool = layer.pool.clone()
        for request, count in enumerate(counts):
            sl = slice(starts[request], starts[request] + count)
            kv = [x.clone() for x in stage_context(cache, table, ends, request, 65536)]
            exact, bf16 = _reference(q[sl], kv, layer.route[sl], layer.used[sl])
            got = out[sl].float()
            err, ref_err = (got - exact).abs(), (bf16.float() - exact).abs()
            # Both round O to BF16; allow one more half-ulp of the largest output.
            assert err.max() <= 1.25 * ref_err.max() + 2**-8 * exact.abs().max()
            assert err.mean() <= 1.25 * ref_err.mean()


def test_only_tiles_leaves_other_rows():
    from aleph_alpha_inference.kernels.nvfp4_attention import nvfp4_attention

    cache, table = _cache([5000])
    rows = 50
    q = torch.randn(rows, 1, 48, 128, device="cuda", dtype=torch.bfloat16)
    route = torch.randint(0, 5000, (rows, 2184), device="cuda").int()
    used = torch.full((rows,), 2184, device="cuda", dtype=torch.int32)
    requests = torch.zeros(rows, device="cuda", dtype=torch.int32)
    full = nvfp4_attention(q, cache, table, requests, route, used)
    out = torch.full_like(q, 7.0)
    flags = torch.tensor([0, 1, 0], device="cuda", dtype=torch.int32)
    nvfp4_attention(
        q, cache, table, requests, route, used, out=out, only_tiles=flags, tile_rows=21
    )
    assert bool((out[:21] == 7).all()) and bool((out[42:] == 7).all())
    torch.testing.assert_close(out[21:42], full[21:42], atol=0, rtol=0)


def test_pages_beyond_64_gib():
    from aleph_alpha_inference.kernels.nvfp4_attention import nvfp4_attention
    from aleph_alpha_inference.kernels.prefill_staging import stage_context
    from aleph_alpha_inference.kernels.summary_cache import SummaryCache

    # Serving caches exceed 2^36 bytes; read pages at their far end.
    pages = (1 << 36) // (PAGE * SummaryCache.STORAGE_BYTES_PER_TOKEN) + 2048
    needed = pages * sum(SummaryCache.page_bytes(PAGE)) + (4 << 30)
    if torch.cuda.mem_get_info()[0] < needed:
        pytest.skip("needs a GPU with room for a 70 GB cache")
    cache = SummaryCache.allocate(pages, PAGE)
    assert cache.storage.numel() > 1 << 36
    length, rows = 4096, 64
    table = (pages - 1 - torch.arange(length // PAGE, device="cuda")).int()[None]
    ids = torch.arange(length, device="cuda")
    k = torch.randn(length, 4, 128, device="cuda", dtype=torch.bfloat16)
    cache.append(k, -k, table[0, ids // PAGE].long() * PAGE + ids % PAGE)
    q = torch.randn(rows, 1, 48, 128, device="cuda", dtype=torch.bfloat16)
    route = torch.randint(0, length, (rows, 2184), device="cuda").int()
    used = torch.full((rows,), 2184, device="cuda", dtype=torch.int32)
    requests = torch.zeros(rows, device="cuda", dtype=torch.int32)
    out = nvfp4_attention(q, cache, table, requests, route, used)
    lengths = torch.tensor([length], device="cuda", dtype=torch.int32)
    kv = [x.clone() for x in stage_context(cache, table, lengths, 0, length)]
    del cache
    exact, bf16 = _reference(q, kv, route, used)
    err, ref_err = (out.float() - exact).abs(), (bf16.float() - exact).abs()
    assert err.max() <= 1.25 * ref_err.max() + 1e-6
