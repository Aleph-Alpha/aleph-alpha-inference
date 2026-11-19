# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
import pytest
import torch


def rounded_nvfp4(x):
    blocks = x.float().unflatten(-1, (-1, 16))
    scales = (blocks.abs().amax(-1) / 6).clamp_min(0.001953125)
    scales = scales.to(torch.float8_e4m3fn).float()
    normalized = (blocks / scales[..., None]).clamp(-6, 6)
    magnitude = normalized.abs()
    step = torch.where(magnitude < 2, 0.5, torch.where(magnitude < 4, 1.0, 2.0))
    rounded = torch.copysign(torch.round(magnitude / step) * step, normalized)
    return (rounded * scales[..., None]).flatten(-2)


def dequant(x, s):
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6], device="cuda"
    )
    ix = torch.stack([x & 15, x >> 4], -1).flatten(-2)
    return (lut[ix.long()] * s.float().repeat_interleave(16, -1)).bfloat16()


@pytest.mark.gpu
def test_fp4_staging_all_values_and_finite_scales():
    from aleph_alpha_inference.kernels.prefill_staging import stage_context, workspace
    from aleph_alpha_inference.kernels.summary_cache import SummaryCache

    cache = SummaryCache.allocate(2)
    values = torch.arange(256, device="cuda", dtype=torch.int32).to(torch.uint8)
    cache.k.copy_(values.reshape(1, 1, 4, 64).expand_as(cache.k))
    cache.v.copy_((255 - values).reshape(1, 1, 4, 64).expand_as(cache.v))
    # All positive finite E4M3 scales, paired with every packed E2M1 byte.
    scales = (torch.arange(256, device="cuda") % 127).to(torch.uint8)
    scales = scales.view(torch.float8_e4m3fn).reshape(2, 128, 1, 1)
    cache.ks.copy_(scales.expand_as(cache.ks))
    cache.vs.copy_(scales.expand_as(cache.vs))
    table = torch.tensor([[1, 0]], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([253], device="cuda", dtype=torch.int32)
    # Rows past the request's length keep what the staging buffer held.
    workspace(cache.storage.device)[:, :256] = float("nan")
    k, v = stage_context(cache, table, lengths, 0, 256)
    indices = torch.cat(
        (torch.arange(128, 256, device="cuda"), torch.arange(128, device="cuda"))
    )
    for actual, raw, scale in ((k, cache.k, cache.ks), (v, cache.v, cache.vs)):
        expected = dequant(raw, scale).flatten(0, 1)[indices]
        torch.testing.assert_close(actual[:253, 0], expected[:253], rtol=0, atol=0)
        assert actual[253:].isnan().all()


@pytest.mark.gpu
def test_fp4_cache_and_attention():
    from aleph_alpha_inference.kernels.sparse_decode import (
        SparseDecodeAttention,
        workspace_shapes,
    )
    from aleph_alpha_inference.kernels.summary_cache import SummaryCache

    torch.manual_seed(42)
    cache = SummaryCache.allocate(4)
    k = torch.randn(512, 4, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    slots = torch.arange(512, device="cuda")
    cache.append(k, v, slots)
    kd = dequant(cache.k, cache.ks)
    vd = dequant(cache.v, cache.vs)
    # Independently nearest-round the original BF16 values using stored scales.
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6], device="cuda"
    )
    for original, raw, scale, decoded in [
        (k, cache.k, cache.ks, kd),
        (v, cache.v, cache.vs, vd),
    ]:
        normalized = original.view(
            4, 128, 4, 128
        ).float() / scale.float().repeat_interleave(16, -1)
        error = (normalized[..., None] - lut).abs().amin(-1)
        actual = (
            normalized - decoded.float() / scale.float().repeat_interleave(16, -1)
        ).abs()
        torch.testing.assert_close(actual, error, atol=0, rtol=0)
    for rows in (1, 7, 64):
        q = torch.randn(rows, 1, 48, 128, device="cuda", dtype=torch.bfloat16)
        table = torch.tensor([[2, 0, 3, 1]], device="cuda", dtype=torch.int32)
        req = torch.zeros(rows, device="cuda", dtype=torch.int32)
        route = torch.stack(
            [torch.randperm(512, device="cuda")[:145] for _ in range(rows)]
        ).int()
        used = torch.arange(rows, device="cuda", dtype=torch.int32) % 145 + 1
        used[-1] = 145
        phys = table[0, route // 128] * 128 + route % 128
        kk = kd.flatten(0, 1)[phys.long()].repeat_interleave(12, 2)
        vv = vd.flatten(0, 1)[phys.long()].repeat_interleave(12, 2)
        logits = (
            torch.einsum("bhd,bkhd->bhk", q[:, 0].float(), kk.float()) * 128**-0.5
        )
        logits.masked_fill_(
            torch.arange(145, device="cuda")[None, None, :] >= used[:, None, None],
            -torch.inf,
        )
        expected = torch.einsum(
            "bhk,bkhd->bhd", logits.softmax(-1), vv.float()
        ).bfloat16()
        attention = SparseDecodeAttention(rows, cache, total=145)
        workspace = tuple(
            torch.empty(shape, device="cuda", dtype=torch.float32)
            for shape in workspace_shapes(rows, attention.splits)
        )
        out = attention(
            q, table, route, used, req, out=torch.empty_like(q), workspace=workspace
        )
        # BF16 tensor core products and different reduction order.
        torch.testing.assert_close(out[:, 0], expected, atol=0.016, rtol=0.016)
