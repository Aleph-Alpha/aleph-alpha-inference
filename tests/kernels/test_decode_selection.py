# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""Exact token selection across coarse-bin boundaries, ties and graph replay."""

import pytest
import torch


def _score_patterns(positions, capacity):
    base = positions.float() / capacity
    return [
        torch.rand_like(base),
        torch.zeros_like(base),
        base,
        1 - base.clamp_max(1),
        positions.remainder(5).float(),
        1 + base * 0.0001,
        1 + (base - 0.96875) * 0.0001,
        base * 1e-7,
        65504 + base * 32,
        1 + base * 2**-19,
        base * torch.finfo(torch.float32).max,
        torch.where(positions.remainder(3) == 0, float("inf"), base),
    ]


@pytest.mark.gpu
def test_reindex_radix_selection_adversarial_graph():
    import cuda.bindings.driver as cuda
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    from aleph_alpha_inference.kernels.decode_selection import RadixSelection

    torch.manual_seed(928)
    batch, width = 4, 16384
    scores = torch.empty(batch, width, device="cuda")
    pool = torch.full((batch, 2048), -1, device="cuda", dtype=torch.int32)
    lengths = torch.tensor([1048576, 8193, 645, 0], device="cuda", dtype=torch.int32)
    remote = torch.empty(batch, 512, device="cuda", dtype=torch.int32)
    route = torch.empty(batch, 644, device="cuda", dtype=torch.int32)
    used = torch.empty(batch, device="cuda", dtype=torch.int32)
    args = [
        from_dlpack(x, assumed_align=4)
        for x in (scores, pool, lengths, remote, route, used)
    ]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    kernel = cute.compile(RadixSelection(mode=2, topk=512), *args, stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        kernel(*args, cuda.CUstream(torch.cuda.current_stream().cuda_stream))
    positions = torch.arange(width, device="cuda")
    patterns = _score_patterns(positions, width)
    for pattern in patterns:
        lengths.copy_(lengths.roll(1))
        pool.fill_(-1)
        for row, length in enumerate(lengths.tolist()):
            blocks = (length + 7) // 8
            count = min(blocks, 2048)
            pool[row, :count] = (
                torch.randperm(blocks, device="cuda")[:count].sort().values.int()
            )
        logical = (
            pool[:, :, None].long() * 8 + torch.arange(8, device="cuda")
        ).flatten(1)
        valid = (logical >= 4) & (logical < lengths[:, None] - 128)
        scores.copy_(pattern[None].expand(batch, -1).masked_fill(~valid, float("nan")))
        graph.replay()
        for row, length in enumerate(lengths.tolist()):
            count = min(length, 644)
            assert int(used[row]) == count
            if length <= 644:
                torch.testing.assert_close(
                    route[row, :count],
                    torch.arange(length, device="cuda", dtype=torch.int32),
                    rtol=0,
                    atol=0,
                )
                continue
            candidates = scores[row].masked_fill(~valid[row], -float("inf"))
            chosen = (
                candidates.argsort(descending=True, stable=True)[:512].sort().values
            )
            expected = logical[row, chosen].int()
            torch.testing.assert_close(remote[row], expected, rtol=0, atol=0)
            expected_route = torch.cat(
                (
                    expected,
                    torch.arange(
                        length - 128, length, device="cuda", dtype=torch.int32
                    ),
                    torch.arange(4, device="cuda", dtype=torch.int32),
                )
            )
            torch.testing.assert_close(route[row], expected_route, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "width,padded", [(2049, False), (4096, True), (32768, True), (65536, False)]
)
def test_pool_radix_scalar_shapes_and_uncached_graph(width, padded):
    import cuda.bindings.driver as cuda
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    from aleph_alpha_inference.kernels.decode_selection import RadixSelection

    torch.manual_seed(934)
    batch = 3
    scores = torch.empty(batch, width + int(padded), device="cuda")[:, :width]
    lengths = torch.tensor([width * 8, 8200, 0], device="cuda", dtype=torch.int32)
    pool = torch.empty(batch, 2048, device="cuda", dtype=torch.int32)
    route = torch.empty(batch, 644, device="cuda", dtype=torch.int32)
    used = torch.empty(batch, device="cuda", dtype=torch.int32)
    args = [
        from_dlpack(tensor, assumed_align=4)
        for tensor in (scores, pool, lengths, pool, route, used)
    ]
    kernel = cute.compile(
        RadixSelection(mode=0, topk=2048),
        *args,
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        kernel(*args, cuda.CUstream(torch.cuda.current_stream().cuda_stream))
    positions = torch.arange(width, device="cuda")
    for pattern in _score_patterns(positions, width):
        lengths.copy_(lengths.roll(1))
        scores.copy_(pattern[None].expand_as(scores))
        graph.replay()
        for row, length in enumerate(lengths.tolist()):
            count = min((length + 7) // 8, 2048)
            expected = (
                scores[row, : (length + 7) // 8]
                .argsort(descending=True, stable=True)[:count]
                .sort()
                .values.int()
            )
            torch.testing.assert_close(pool[row, :count], expected, rtol=0, atol=0)
            assert bool((pool[row, count:] == -1).all())


@pytest.mark.gpu
def test_boundary_cardinalities_graph():
    import cuda.bindings.driver as cuda
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    from aleph_alpha_inference.kernels.decode_selection import RadixSelection

    torch.manual_seed(957)
    counts = [1, 31, 32, 33, 63, 64, 65, 127, 128, 129]
    batch, width = len(counts), 8192
    scores = torch.zeros(batch, width, device="cuda")
    lengths = torch.full((batch,), width * 8, device="cuda", dtype=torch.int32)
    out = torch.empty(batch, 2048, device="cuda", dtype=torch.int32)
    route = torch.empty(batch, 644, device="cuda", dtype=torch.int32)
    used = torch.empty(batch, device="cuda", dtype=torch.int32)
    args = [
        from_dlpack(tensor, assumed_align=4)
        for tensor in (scores, out, lengths, out, route, used)
    ]
    fn = cute.compile(
        RadixSelection(mode=0, topk=2048),
        *args,
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn(*args, cuda.CUstream(torch.cuda.current_stream().cuda_stream))
    for trial in range(6):
        for row, count in enumerate(counts):
            remaining = count // 2 + 1
            above = 2048 - remaining
            values = torch.full((width,), 0.25, device="cuda")
            values[:above] = 2
            # These distinct FP32 values share one coarse histogram bucket.
            # Alternate ties and distinct keys, with new logical IDs per replay.
            increment = 0 if trial % 2 else 2**-20
            values[above : above + count] = (
                1.001 + torch.arange(count, device="cuda").float() * increment
            )
            scores[row] = values[torch.randperm(width, device="cuda")]
        out.fill_(-1)
        graph.replay()
        reference = (
            scores.argsort(dim=1, descending=True, stable=True)[:, :2048]
            .sort(dim=1)
            .values.int()
        )
        torch.testing.assert_close(out, reference, rtol=0, atol=0)
