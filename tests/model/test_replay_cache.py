# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""Bounded replay preserves FA4 tiling and cannot read poisoned prefix values."""

from types import SimpleNamespace

import cutlass.cute as cute

import pytest
import torch
from aleph_alpha_inference.model import replay_cache
from aleph_alpha_inference.model.replay_cache import (
    clear_boundary,
    replay_cache_alignment,
)

from tensor_asserts import assert_equal

# The tests import `vllm_swa` and `fa_utils.flash_attn_varlen_func` themselves:
# it needs vLLM's CUDA flash attention extensions, which the CPU-only CI runner
# lacks, so importing it here would fail collection there.

pytestmark = pytest.mark.gpu


def test_boundary_clear_uses_64bit_physical_cache_offsets():
    page_size, page_stride = 16, 143872
    first_high_page = (2**31 + page_stride - 1) // page_stride
    ids = [0, 1, *range(first_high_page, first_high_page + 8)]
    cache = torch.empty_strided(
        (ids[-1] + 1, 4, page_size, 256),
        (page_stride, 4096, 256, 1),
        device="cuda",
        dtype=torch.uint8,
    )
    # The allocation matches vLLM's interleaved FP8 SWA cache. Initialize only
    # the pages read below, including guards that must remain unchanged.
    for page in ids:
        cache[page].fill_(127)
    table = torch.tensor([ids[2:]], device="cuda", dtype=torch.int32)
    limits = torch.tensor([127], device="cuda", dtype=torch.int32)
    k, v = cache.transpose(1, 2).split(128, dim=-1)
    clear_boundary(k, v, table, limits)
    actual = cache[torch.tensor(ids, device="cuda")]
    expected = torch.full_like(actual, 127)
    expected[2:-1] = 0
    expected[-1, :, :15] = 0
    assert_equal(actual, expected)


def run_replay(q, cache, table, lengths, starts, floors, fp8, splits=1):
    from aleph_alpha_inference.model.vllm_swa import CuteSWAImpl

    metadata = SimpleNamespace(
        num_actual_tokens=q.size(0),
        max_num_splits=splits,
        block_table=table,
        query_start_loc=torch.tensor(starts, device="cuda", dtype=torch.int32),
        max_query_len=max(b - a for a, b in zip(starts, starts[1:])),
        seq_lens=torch.tensor(lengths, device="cuda", dtype=torch.int32),
        max_seq_len=max(lengths),
        replay_floor=torch.tensor(floors, device="cuda", dtype=torch.int32),
    )
    byte_cache = cache.view(torch.uint8) if fp8 else cache
    impl = object.__new__(CuteSWAImpl)
    impl.head_size, impl.num_kv_heads = 128, 4
    impl.scale, impl.sliding_window = 128**-0.5, (512, 0)
    impl.kv_cache_dtype = "fp8_e4m3" if fp8 else "bfloat16"
    impl.supports_quant_query_input = fp8
    layer = SimpleNamespace(
        _q_scale=torch.tensor(0.125 if fp8 else 1.0, device="cuda"),
        _k_scale=torch.tensor(0.0625 if fp8 else 1.0, device="cuda"),
        _v_scale=torch.tensor(0.25 if fp8 else 1.0, device="cuda"),
    )
    out = torch.empty(q.shape, device="cuda", dtype=torch.bfloat16)
    return impl.forward(layer, q, None, None, byte_cache, metadata, out)


def check_replay(prefixes, query_lengths, page, fp8):
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func
    from vllm.vllm_flash_attn.cute import utils

    @cute.jit
    def reference_mask(batch, head, query, key, seqlens, aux):
        return key >= aux[0][utils.ssa_to_scalar(batch)]

    torch.manual_seed(713)
    lengths = [p + 513 for p in prefixes]
    pages = [(n + page - 1) // page for n in lengths]
    starts = [0]
    for n in query_lengths:
        starts.append(starts[-1] + n)
    q = torch.randn(starts[-1], 48, 128, device="cuda", dtype=torch.bfloat16)
    cache = torch.randn(sum(pages), 4, page, 256, device="cuda", dtype=q.dtype)
    scales = (
        dict(
            q_descale=torch.full((len(prefixes), 4), 0.125, device="cuda"),
            k_descale=torch.full((len(prefixes), 4), 0.0625, device="cuda"),
            v_descale=torch.full((len(prefixes), 4), 0.25, device="cuda"),
        )
        if fp8
        else {}
    )
    if fp8:
        q = (q / 0.125).to(torch.float8_e4m3fn)
        cache[..., :128] /= 0.0625
        cache[..., 128:] /= 0.25
        cache = cache.to(torch.float8_e4m3fn)
    k, v = cache.transpose(1, 2).split(128, dim=-1)
    permutation = torch.randperm(sum(pages), device="cuda", dtype=torch.int32)
    table = torch.zeros(len(prefixes), max(pages), device="cuda", dtype=torch.int32)
    begin = 0
    for r, n in enumerate(pages):
        table[r, :n] = permutation[begin : begin + n]
        begin += n

    # Independent native FA4 reference keeps absolute positions and 128-key
    # tile boundaries. Physically truncating at an unaligned floor changes
    # FP8 P rounding even between two native calls, so is not an exact oracle.
    def ordered(t):
        raw = t.view(torch.uint8) if fp8 else t
        pieces = []
        for r, (prefix, length, num_pages) in enumerate(zip(prefixes, lengths, pages)):
            selected = (
                raw[table[r, :num_pages].long()].reshape(-1, 4, 128)[:length].clone()
            )
            selected[:prefix] = 0
            pieces.append(selected)
        return torch.cat(pieces).view(t.dtype)

    kr, vr = ordered(k), ordered(v)
    original = cache.view(torch.uint8).clone()
    bases = [
        p // replay_cache_alignment(page) * replay_cache_alignment(page)
        for p in prefixes
    ]
    relative_lengths = [n - base for n, base in zip(lengths, bases)]
    cut_table = torch.zeros(
        len(prefixes),
        max((n + page - 1) // page for n in relative_lengths),
        device="cuda",
        dtype=torch.int32,
    )
    poison_prefix(k, v, table, cut_table, prefixes, bases, pages, page, fp8)
    actual = run_replay(
        q,
        cache,
        cut_table,
        relative_lengths,
        starts,
        [p - base for p, base in zip(prefixes, bases)],
        fp8,
    )
    k_starts = [0]
    for n in lengths:
        k_starts.append(k_starts[-1] + n)
    expected = flash_attn_varlen_func(
        q=q,
        k=kr,
        v=vr,
        cu_seqlens_q=torch.tensor(starts, device="cuda", dtype=torch.int32),
        max_seqlen_q=max(query_lengths),
        cu_seqlens_k=torch.tensor(k_starts, device="cuda", dtype=torch.int32),
        max_seqlen_k=max(lengths),
        causal=True,
        window_size=(512, 0),
        fa_version=4,
        num_splits=1,
        out=torch.empty_like(actual),
        mask_mod=reference_mask,
        aux_tensors=[torch.tensor(prefixes, device="cuda", dtype=torch.int32)],
        **scales,
    )
    assert_equal(actual, expected)
    check_cache_bytes(cache, original, table, prefixes, bases, lengths, page)


def poison_prefix(k, v, table, cut_table, prefixes, bases, pages, page, fp8):
    for r, (prefix, base, num_pages) in enumerate(zip(prefixes, bases, pages)):
        ids = torch.arange(prefix, device="cuda")
        phys = table[r, ids // page].long()
        if fp8:
            # E4M3 0x7f is NaN; clear must operate on the actual byte cache.
            k.view(torch.uint8)[phys, ids % page] = 127
            v.view(torch.uint8)[phys, ids % page] = 127
        else:
            k[phys, ids % page] = float("nan")
            v[phys, ids % page] = float("nan")
        count = num_pages - base // page
        cut_table[r, :count] = table[r, base // page : num_pages]


def check_cache_bytes(cache, original, table, prefixes, bases, lengths, page):
    k, v = cache.transpose(1, 2).split(128, dim=-1)
    # Clearing may touch only the rebased, skipped prefix; retained KV bytes
    # and completely excluded pages must stay unchanged.
    for r, (prefix, base, length) in enumerate(zip(prefixes, bases, lengths)):
        ids = torch.arange(base, prefix, device="cuda")
        phys = table[r, ids // page].long()
        assert torch.count_nonzero(k.view(torch.uint8)[phys, ids % page]) == 0
        assert torch.count_nonzero(v.view(torch.uint8)[phys, ids % page]) == 0
        ids = torch.arange(prefix, length, device="cuda")
        phys = table[r, ids // page].long()
        assert_equal(
            cache.view(torch.uint8)[phys, :, ids % page], original[phys, :, ids % page]
        )
        ids = torch.arange(base, device="cuda")
        phys = table[r, ids // page].long()
        assert torch.isnan(k.float()[phys, ids % page]).all()
        assert torch.isnan(v.float()[phys, ids % page]).all()


@pytest.mark.parametrize("prefix", [1535, 1536, 1537])
@pytest.mark.parametrize("page", [16, 128])
@pytest.mark.parametrize("fp8", [False, True])
def test_replay_swa_matches_native_reference(prefix, page, fp8):
    check_replay([prefix], [137], page, fp8)


@pytest.mark.parametrize("page", [16, 128])
@pytest.mark.parametrize("fp8", [False, True])
def test_replay_packed_prefill_and_decode(page, fp8):
    check_replay([1535, 1537, 0], [137, 1, 65], page, fp8)


@pytest.mark.parametrize("query_length", [1, 17, 51])
@pytest.mark.parametrize("page", [16, 128])
def test_replay_preserves_native_fp8_split_policy(query_length, page):
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

    torch.manual_seed(719)
    length = 16384 + query_length
    floor = length - 8193
    base = floor // replay_cache_alignment(page) * replay_cache_alignment(page)
    pages = (length + page - 1) // page
    q = torch.randn(query_length, 48, 128, device="cuda", dtype=torch.bfloat16)
    q = (q / 0.125).to(torch.float8_e4m3fn)
    cache = torch.randn(pages, 4, page, 256, device="cuda", dtype=torch.bfloat16)
    cache[..., :128] /= 0.0625
    cache[..., 128:] /= 0.25
    cache = cache.to(q.dtype)
    table = torch.randperm(pages, device="cuda", dtype=torch.int32)[None]
    k, v = cache.transpose(1, 2).split(128, dim=-1)
    expected = flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=torch.tensor([0, query_length], device="cuda", dtype=torch.int32),
        max_seqlen_q=query_length,
        seqused_k=torch.tensor([length], device="cuda", dtype=torch.int32),
        max_seqlen_k=length,
        block_table=table,
        softmax_scale=128**-0.5,
        causal=True,
        window_size=(512, 0),
        fa_version=4,
        num_splits=0,
        q_descale=torch.full((1, 4), 0.125, device="cuda"),
        k_descale=torch.full((1, 4), 0.0625, device="cuda"),
        v_descale=torch.full((1, 4), 0.25, device="cuda"),
        out=torch.empty(q.shape, device="cuda", dtype=torch.bfloat16),
    )
    actual = run_replay(
        q,
        cache,
        table[:, base // page :],
        [length - base],
        [0, query_length],
        [floor - base],
        True,
        splits=0,
    )
    assert_equal(actual, expected)


def test_boundary_clear_reuses_code_across_request_counts_and_widths():
    replay_cache._COMPILED.clear()
    cache = torch.full((64, 4, 16, 256), 7, device="cuda", dtype=torch.uint8)
    k, v = cache.transpose(1, 2).split(128, dim=-1)
    for requests, width in ((1, 3), (5, 2), (2, 7), (5, 2)):
        cache.fill_(7)
        table = torch.randperm(64, device="cuda").int()[: requests * width]
        table = table.view(requests, width)
        limits = torch.randint(0, 16 * width, (requests,), device="cuda").int()
        replay_cache.clear_boundary(k, v, table, limits)
        expected = torch.full_like(cache, 7)
        for row in range(requests):
            for token in range(int(limits[row])):
                expected[table[row, token // 16].long(), :, token % 16] = 0
        assert_equal(cache, expected)
    assert len(replay_cache._COMPILED) == 1
