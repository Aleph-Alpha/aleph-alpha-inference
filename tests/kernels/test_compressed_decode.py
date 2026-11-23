# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""The extra policies retain fixed decode buffers and support CUDA graphs."""

import pytest
import torch


@pytest.mark.gpu
@pytest.mark.parametrize("compression,pool_tokens", [(16, 16384), (16, 32768)])
def test_compressed_decode_graph(compression, pool_tokens):
    from aleph_alpha_inference.kernels.summary_cache import SummaryCache

    from serving_arena import (
        decode_scratch,
        new_owner,
        served_layer,
    )

    torch.manual_seed(8)
    lengths = [65539, 33793]
    pages = [(n + 127) // 128 for n in lengths]
    cache = SummaryCache.allocate(sum(pages), 128)
    table = torch.zeros(2, max(pages), device="cuda", dtype=torch.int32)
    page = 0
    for i, n in enumerate(lengths):
        table[i, : pages[i]] = torch.arange(page, page + pages[i], device="cuda")
        key = torch.randn(n, 4, 128, device="cuda", dtype=torch.bfloat16)
        value = torch.randn_like(key)
        slots = torch.arange(n, device="cuda") + page * 128
        cache.append(key[:-3], value[:-3], slots[:-3])
        for j in range(n - 3, n):
            cache.append(key[j : j + 1], value[j : j + 1], slots[j : j + 1])
        page += pages[i]
    q = torch.randn(2, 1, 48, 128, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(q)
    cu = torch.arange(3, device="cuda", dtype=torch.int32)
    ends = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    assert compression == SummaryCache.BLOCK
    runtime = served_layer(
        cache,
        table,
        2,
        max(lengths),
        2048,
        True,
        owner=new_owner(),
        source=True,
        pool_tokens=pool_tokens,
    )
    step = dict(role="source", out=out, query_starts=())
    for _ in range(2):
        runtime.run(
            q, cu, ends, bounds=[], attention_workspace=decode_scratch(runtime), **step
        )
    expected = [t.clone() for t in (out, runtime.pool, runtime.route, runtime.used)]
    assert (runtime.used == 2184).all()
    assert (runtime.route >= 0).all()
    assert (runtime.route < ends[:, None]).all()
    scratch = decode_scratch(runtime)
    runtime.run(q, cu, ends, bounds=[], attention_workspace=scratch, **step)
    torch.testing.assert_close(out, expected[0], rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        runtime.run(q, cu, ends, bounds=[], attention_workspace=scratch, **step)
    graph.replay()
    for got, ref in zip((out, runtime.pool, runtime.route, runtime.used), expected):
        torch.testing.assert_close(got, ref, rtol=0, atol=0)
