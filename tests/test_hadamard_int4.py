# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""Hadamard int4 KV cache: config, the QK fold, page layout against fragment reads,
round trips, serving."""

import math
import os
from types import SimpleNamespace

import pytest
import torch

from aleph_alpha_inference import hadamard_int4 as h4
from aleph_alpha_inference.hadamard_int4 import page


def _rel(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return (
        (actual.float().flatten() - expected.float().flatten()).norm()
        / expected.float().flatten().norm()
    ).item()


def test_config_and_block_size():
    def parse(extra):
        return h4.get_config(SimpleNamespace(additional_config=extra))

    assert parse({}) is None
    assert parse({"kolibri1": {"kv_cache": "hadamard-int4"}}).seed == h4.DEFAULT_SEED
    cfg = parse({"kolibri1": {"kv_cache": "hadamard-int4", "hadamard_seed": 7}})
    assert cfg.seed == 7
    with pytest.raises(ValueError):
        parse({"kolibri1": {"kv_cache": "int3"}})
    with pytest.raises(ValueError):
        parse({"kolibri1": {"kv_cache": "hadamard-int4", "bits": 3}})
    # 32 int4 tokens (32 * 136 B) cover one 16-token fp8 page (16 * 256 B);
    # 64 a bf16 one
    assert h4.full_attention_block_size(256) == 32
    assert h4.full_attention_block_size(512) == 64
    assert h4.full_attention_block_size(256, min_block=48) == 64


def test_rotation():
    first, again = h4.layer_rotation(2), h4.layer_rotation(2)
    other = h4.layer_rotation(2, seed=5)
    assert torch.equal(first, again)
    assert not torch.equal(first, other)
    assert torch.allclose(first @ first.T, torch.eye(128), atol=1e-5)
    # built on the CPU, so every rank of a distributed run draws the same matrix
    assert h4.layer_rotation(2).device.type == "cpu"


def test_fold_preserves_scores():
    """(n_q @ M_q) . (n_k @ M_k) == (g_q n_q) . (g_k n_k): the stock scores."""
    gen = torch.Generator().manual_seed(0)
    q_gain = 0.5 + 2 * torch.rand(128, generator=gen)
    k_gain = 0.5 + 2 * torch.rand(128, generator=gen)
    nq, nk = torch.randn(64, 128, generator=gen), torch.randn(80, 128, generator=gen)
    q_map, k_map = h4.fold_norm_gains(q_gain, k_gain, h4.layer_rotation(0))
    reference = (nq * q_gain) @ (nk * k_gain).T
    got = (nq @ q_map.float()) @ (nk @ k_map.float()).T
    assert _rel(got, reference) < 2e-3


def test_page_tables_partition():
    """The key tables place each of the 128 dims once; each of the 16 value token slots
    covers all 128 dims with its 64 bytes, and the slots partition the
    1024-byte value block."""
    layout = page.tables(torch.device("cpu"))
    # key tables: each of the 128 dims appears exactly once across the 64 bytes
    assert sorted(
        layout["k_lo"].long().tolist() + layout["k_hi"].long().tolist()
    ) == list(range(128))
    # value tables: for each of the 16 token slots, the slot owns 64 distinct bytes
    # whose low and high nibbles cover all 128 dims once.
    for ts in range(16):
        pos = layout["v_pos"][ts].long().tolist()
        lo = layout["v_lo"][ts].long().tolist()
        hi = layout["v_hi"][ts].long().tolist()
        assert len(pos) == 64
        dims = lo + hi
        assert sorted(dims) == list(range(128))
    # across the 16 slots, the bytes partition the 1024-byte block with no byte shared.
    all_pos = []
    for ts in range(16):
        all_pos.extend(layout["v_pos"][ts].long().tolist())
    assert sorted(all_pos) == list(range(1024))


@pytest.mark.gpu
@pytest.mark.parametrize("T", [1, 300])
def test_qk_rotate(T):
    from aleph_alpha_inference.hadamard_int4.qk_rotate import (
        rotate_qk,
        rotate_qk_reference,
    )

    gen = torch.Generator().manual_seed(T)
    HQ, HKV = 24, 2
    W = (HQ + 2 * HKV) * 128
    # strided rows, like a slice of a wider GEMM output
    qkv = (torch.randn(T, W + 64, generator=gen) * 3).cuda().bfloat16()[:, :W]
    g_q = (0.5 + 2 * torch.rand(128, generator=gen)).cuda()
    g_k = (0.5 + 2 * torch.rand(128, generator=gen)).cuda()
    rotation = h4.layer_rotation(3)
    q_map, k_map = h4.fold_norm_gains(g_q, g_k, rotation)
    q, k = rotate_qk(qkv, q_map, k_map, HQ, HKV, 1e-6)
    rq, rk = rotate_qk_reference(qkv, g_q, g_k, rotation.cuda(), HQ, HKV, 1e-6)
    assert _rel(q, rq) < 4e-3 and _rel(k, rk) < 4e-3


@pytest.mark.gpu
def test_pack_restore_decode():
    from aleph_alpha_inference.hadamard_int4 import decode, page, store

    gen = torch.Generator().manual_seed(0)
    B, L, HKV, G, bs = 3, 1000, 2, 12, 32
    pages = -(-L // bs)
    bt = torch.randperm(B * pages, generator=gen).view(B, pages).int().cuda()
    cache = page.alloc_cache(B * pages, HKV, bs, "cuda")
    tok = torch.arange(L, device="cuda")
    ks = (torch.randn(B, L, HKV, 128, generator=gen) + 1.5).cuda().bfloat16()
    vs = torch.randn(B, L, HKV, 128, generator=gen).cuda().bfloat16()
    for b in range(B):
        slots = (bt[b, tok // bs] * bs + tok % bs).int()
        store.pack(ks[b], vs[b], slots, cache)
    lens = torch.tensor([1000, 777, 1], dtype=torch.int32, device="cuda")
    cu = torch.cat([lens.new_zeros(1), lens.cumsum(0)]).int()
    kpack, vpack = store.restore(cache, bt, cu, L)
    # attention kernels read whole tiles past the packed end: those rows must be finite
    tail = slice(int(cu[-1]), int(cu[-1]) + store.RESTORE_PAD)
    assert not kpack[tail].any() and not vpack[tail].any()
    q = torch.randn(B, G * HKV, 128, generator=gen).cuda().bfloat16()
    out = decode.decode(q, cache, bt, lens)
    assert torch.equal(
        out, decode.decode(q.half(), cache, bt, lens)
    )  # in-kernel bf16 -> fp16
    for b in range(B):
        n, a = int(lens[b]), int(cu[b])
        kd, vd = store.restore_reference(cache, bt[b], n)
        # rare one-code differences on rounding boundaries are fine;
        # a layout bug scrambles most
        assert (
            (kd - store.pack_reference(ks[b, :n])).abs() > 0.05
        ).float().mean() < 1e-3
        assert (
            (vd - store.pack_reference(vs[b, :n])).abs() > 0.05
        ).float().mean() < 1e-3
        assert _rel(kpack[a : a + n], kd) < 1e-2 and _rel(vpack[a : a + n], vd) < 1e-2
        p = torch.einsum(
            "hd,lhd->hl", q[b].float(), kd.repeat_interleave(G, 1)
        ) / math.sqrt(128)
        ref = torch.einsum("hl,lhd->hd", p.softmax(-1), vd.repeat_interleave(G, 1))
        assert _rel(out[b], ref) < 1e-2


@pytest.fixture(scope="module")
def kolibri1_hd128_dir(tmp_path_factory):
    """Tiny checkpoint with Kolibri's head_dim, QK-norm gains != 1, and weights
    scaled by 1/sqrt(fan_in) so the model is smooth enough for logprob comparisons."""
    from checkpoints import KOLIBRI1_CONFIG, write_kolibri1_checkpoint
    from safetensors.torch import load_file, save_file

    cfg = dict(
        KOLIBRI1_CONFIG, head_dim=128, num_attention_heads=8, num_key_value_heads=2
    )
    d = write_kolibri1_checkpoint(str(tmp_path_factory.mktemp("kolibri1_hd128")), cfg)
    path = os.path.join(d, "model.safetensors")
    t = load_file(path)
    g = torch.Generator().manual_seed(0)
    for name, w in t.items():
        if name.endswith(("q_norm.weight", "k_norm.weight")):
            t[name] = 0.5 + 2 * torch.rand(128, generator=g)
        elif w.dim() == 2 and "embed_tokens" not in name:
            t[name] = w / math.sqrt(w.shape[1])
    save_file(t, path)
    return d


@pytest.mark.gpu
def test_serving_matches_stock(register_plugin, kolibri1_hd128_dir, tokenizer):
    """Chunked prefill (FlashAttention on the restored history) and decode (CuTe kernel
    in CUDA graphs): the Hadamard int4 cache deviates from bf16 KV about as much
    as stock fp8 KV."""
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    g = torch.Generator().manual_seed(1)
    prompts = [
        TokensPrompt(
            prompt_token_ids=torch.randint(0, 50000, (n,), generator=g).tolist()
        )
        for n in (700, 40, 1500)
    ]

    def logprobs(kv, **kw):
        llm = LLM(
            model=kolibri1_hd128_dir,
            tokenizer=tokenizer,
            max_model_len=2048,
            seed=0,
            gpu_memory_utilization=0.25,
            kv_cache_dtype=kv,
            max_num_batched_tokens=512,
            **kw,
        )
        outs = llm.generate(
            prompts,
            SamplingParams(max_tokens=8, temperature=0, prompt_logprobs=0),
            use_tqdm=False,
        )
        assert all(len(o.outputs[0].token_ids) == 8 for o in outs)
        return torch.tensor(
            [
                next(iter(d.values())).logprob
                for o in outs
                for d in o.prompt_logprobs[1:]
            ]
        )

    ref = logprobs("auto")
    noise = (logprobs("fp8") - ref).abs().mean().item()
    dev = (
        (
            logprobs(
                "fp8", additional_config={"kolibri1": {"kv_cache": "hadamard-int4"}}
            )
            - ref
        )
        .abs()
        .mean()
    )
    assert dev.item() < 3 * max(noise, 0.01), (dev.item(), noise)


@pytest.mark.gpu
def test_serving_prefix_cache(register_plugin, kolibri1_hd128_dir, tokenizer):
    """A repeated prompt served from the prefix cache (int4 full-attention blocks of 32
    tokens next to 16-token sliding blocks) decodes exactly like its first, uncached
    run."""
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    g = torch.Generator().manual_seed(2)
    ids = torch.randint(0, 50000, (1000,), generator=g).tolist()
    llm = LLM(
        model=kolibri1_hd128_dir,
        tokenizer=tokenizer,
        max_model_len=2048,
        seed=0,
        gpu_memory_utilization=0.25,
        kv_cache_dtype="fp8",
        enable_prefix_caching=True,
        additional_config={"kolibri1": {"kv_cache": "hadamard-int4"}},
    )
    sp = SamplingParams(max_tokens=16, temperature=0, logprobs=0)
    first, again = (
        llm.generate([TokensPrompt(prompt_token_ids=ids)], sp, use_tqdm=False)[0]
        for _ in range(2)
    )
    assert again.num_cached_tokens > 0
    assert first.outputs[0].token_ids == again.outputs[0].token_ids
    lp = [
        [next(iter(d.values())).logprob for d in o.outputs[0].logprobs]
        for o in (first, again)
    ]
    # cached and uncached prefill batch the work differently:
    # logprobs agree, not bitwise
    assert torch.allclose(torch.tensor(lp[0]), torch.tensor(lp[1]), atol=0.05)
