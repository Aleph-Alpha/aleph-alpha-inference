# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""Compressed summaries agree across bulk and incremental cache appends."""

import pytest
import torch


def reference_pack(values):
    groups = values.reshape(-1, 8, 16)
    scales = (groups.abs().amax(-1) / 6).clamp(min=2**-9).to(torch.float8_e4m3fn)
    normalized = groups / scales.float()[..., None]
    levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=values.device)
    distance = (normalized.abs()[..., None] - levels).abs()
    tied = distance == distance.amin(-1, keepdim=True)
    # Nearest-even E2M1 encoding, including signed zero.
    order = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], device=values.device)
    code = torch.where(tied, order, 99).argmin(-1).to(torch.uint8)
    code |= torch.signbit(normalized).to(torch.uint8) * 8
    code = code.reshape(-1, 128)
    return code[:, ::2] | (code[:, 1::2] << 4), scales.view(torch.uint8)


@pytest.mark.gpu
def test_summary_reference_and_incremental_append():
    from aleph_alpha_inference.kernels.summary_cache import SummaryCache

    cache_type = SummaryCache
    compression = cache_type.BLOCK
    torch.manual_seed(83)
    lengths = [137, 257, 19]
    n = sum(lengths)
    k = torch.randn(n, 4, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    pages = sum((length + 127) // 128 for length in lengths)
    whole = cache_type.allocate(pages, 128)
    chunks = cache_type.allocate(pages, 128)
    # Main K/V, routing keys, summaries and partial sums, without padding.
    assert cache_type.page_bytes(128) == (73728, 9216, 576 * 16 // compression, 512)
    offset = page = 0
    for length in lengths:
        slots = torch.arange(length, device="cuda") + page * 128
        keys, values = k[offset : offset + length], v[offset : offset + length]
        whole.append(keys, values, slots)
        # Cross block/page boundaries, then append the tail one token at a time.
        boundaries = sorted(
            set([0, min(3, length), min(129, length), max(0, length - 19), length])
        )
        for first, last in zip(boundaries[:-1], boundaries[1:]):
            if first >= length - 19:
                for i in range(first, last):
                    chunks.append(keys[i : i + 1], values[i : i + 1], slots[i : i + 1])
            else:
                chunks.append(keys[first:last], values[first:last], slots[first:last])
        count = length // compression
        means = (
            keys[: count * compression, 0]
            .float()
            .reshape(count, compression, 128)
            .sum(1)
            / compression
        )
        expected, expected_scales = reference_pack(means)
        indices = torch.arange(count, device="cuda")
        pg, block = page + indices // (128 // compression), indices % (
            128 // compression
        )
        torch.testing.assert_close(whole.summaries[pg, block], expected, rtol=0, atol=0)
        torch.testing.assert_close(
            whole.summary_scales[pg, block].view(torch.uint8),
            expected_scales,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            chunks.summaries[pg, block], whole.summaries[pg, block], rtol=0, atol=0
        )
        torch.testing.assert_close(
            chunks.summary_scales[pg, block],
            whole.summary_scales[pg, block],
            rtol=0,
            atol=0,
        )
        offset += length
        page += (length + 127) // 128
