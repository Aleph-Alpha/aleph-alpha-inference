# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""The cache append gives the same cache however a step presents its rows."""

import pytest
import torch


def _zeroed(pages):
    from aleph_alpha_inference.kernels.summary_cache import SummaryCache

    cache = SummaryCache.allocate(pages, 128)
    for tensor in cache.tensors:
        tensor.zero_()
    return cache


@pytest.mark.gpu
def test_batched_append_matches_per_request_appends():
    torch.manual_seed(11)
    dev = torch.device("cuda")
    lengths = [1, 37, 300, 129, 1000]
    pages = [(n + 127) // 128 + 1 for n in lengths]
    batched, reference = (_zeroed(sum(pages)) for _ in range(2))
    order = torch.randperm(sum(pages), device=dev)
    tables, first = [], 0
    for count in pages:
        tables.append(order[first : first + count])
        first += count
    written = [0] * len(lengths)
    # Rounds of prefill chunks, single decode tokens and spec-decode bursts,
    # batched across requests with padding slots, starting at any offset.
    for step, sizes in enumerate(
        ([1, 5, 77, 0, 300], [0, 1, 1, 64, 3], [0, 31, 100, 65, 697], [0, 0, 22, 0, 0])
    ):
        slots = []
        for r, size in enumerate(sizes):
            size = min(size, lengths[r] - written[r])
            pos = torch.arange(written[r], written[r] + size, device=dev)
            slots.append(tables[r][pos // 128] * 128 + pos % 128)
            written[r] += size
        slots.insert(1, torch.full((3,), -1, device=dev, dtype=torch.long))
        counts = [s.numel() for s in slots]
        slots = torch.cat(slots)
        # Rows of a fused QKV projection: strided K/V views.
        qkv = torch.randn(slots.numel(), 6144 + 1024, device=dev, dtype=torch.bfloat16)
        k = qkv[:, 6144:6656].view(-1, 4, 128)
        v = qkv[:, 6656:].view(-1, 4, 128)
        batched.append(k, v, slots)
        # The reference appends each request's rows alone, as contiguous
        # copies, and never sees the padding rows.
        start = 0
        for i, count in enumerate(counts):
            rows = slice(start, start + count)
            start += count
            if i != 1 and count:
                reference.append(k[rows].clone(), v[rows].clone(), slots[rows])
        for got, ref in zip(batched.tensors, reference.tensors):
            torch.testing.assert_close(got, ref, rtol=0, atol=0, msg=f"step {step}")


@pytest.mark.gpu
def test_append_rejects_unaligned_rows():
    cache = _zeroed(2)
    raw = torch.randn(33 * 512 + 8, device="cuda", dtype=torch.bfloat16)
    k = raw[8:].view(33, 4, 128)
    slots = torch.arange(33, device="cuda")
    with pytest.raises(RuntimeError, match="32-byte aligned K/V rows"):
        cache.append(k, k, slots)
