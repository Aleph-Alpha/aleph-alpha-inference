# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""Decode runtimes sharing one routing arena match runtimes on fresh arenas exactly."""

from types import SimpleNamespace

import pytest
import torch

from sparse_hf_config import sparse_hf_config

pytestmark = pytest.mark.gpu

SPAN = 98304
TOPK = 2048


def _cache(lengths):
    from aleph_alpha_inference.kernels.summary_cache import SummaryCache

    pages = [(n + 127) // 128 for n in lengths]
    cache = SummaryCache.allocate(sum(pages), 128)
    table = torch.zeros(len(lengths), SPAN // 128, device="cuda", dtype=torch.int32)
    page = 0
    for i, n in enumerate(lengths):
        table[i, : pages[i]] = torch.arange(page, page + pages[i], device="cuda")
        key = torch.randn(n, 4, 128, device="cuda", dtype=torch.bfloat16)
        cache.append(
            key, torch.randn_like(key), torch.arange(n, device="cuda") + page * 128
        )
        page += pages[i]
    return cache, table


def _impls(monkeypatch, compression, cache):
    import aleph_alpha_inference.kernels.vllm_backend as backend
    from vllm.v1.worker.workspace import WorkspaceManager

    config = SimpleNamespace(
        model_config=SimpleNamespace(
            max_model_len=SPAN,
            hf_text_config=sparse_hf_config(
                index_compression=compression, index_topk=TOPK, index_pool_size=16384
            ),
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=8, max_num_batched_tokens=8192),
        parallel_config=SimpleNamespace(use_ubatching=False),
        compilation_config=SimpleNamespace(max_cudagraph_capture_size=8),
    )
    monkeypatch.setattr(backend, "get_current_vllm_config", lambda: config)
    manager = WorkspaceManager(torch.device("cuda"), num_lanes=1)
    monkeypatch.setattr(backend, "current_workspace_manager", lambda: manager)
    source = backend.Block8Impl(48, 128, 128**-0.5, num_kv_heads=4)
    reindex = backend.Block8Impl(48, 128, 128**-0.5, num_kv_heads=4)
    source.sparse_role, reindex.sparse_role = "indexer", "reindexer"
    reindex.source_attention = SimpleNamespace(impl=source)
    active = [None]
    monkeypatch.setattr(
        backend,
        "get_attention_context",
        lambda _: (active[0], None, cache.storage, None),
    )
    for impl in (source, reindex):
        monkeypatch.setattr(impl, "_cache_view", lambda _: cache)
    return source, reindex, active


def _step(source, reindex, active, cache, table, lengths, q, reindex_q):
    rows = q.shape[0]
    active[0] = SimpleNamespace(
        num_actual_tokens=rows,
        max_query_len=1,
        max_seq_len=SPAN,
        query_start_loc=torch.arange(rows + 1, device="cuda", dtype=torch.int32),
        seq_lens=lengths,
        block_table=table,
        row_bounds=[],
        query_starts=(),
        min_context=0,
    )
    outputs = []
    for impl, query in ((source, q), (reindex, reindex_q)):
        out = torch.full_like(query, float("nan"))
        impl.forward(
            SimpleNamespace(layer_name="attn"),
            query[:, 0],
            None,
            None,
            cache.storage,
            None,
            out[:, 0],
        )
        outputs.append(out)
    return outputs


def _reference(cache, table, lengths, q, reindex_q):
    from serving_arena import (
        decode_scratch,
        new_owner,
        served_layer,
    )

    rows = q.shape[0]
    cu = torch.arange(rows + 1, device="cuda", dtype=torch.int32)
    outputs = []
    pool, owner = None, new_owner()
    for role, query in (("source", q), ("reindex", reindex_q)):
        runtime = served_layer(
            cache, table, rows, SPAN, TOPK, True, owner=owner, source=role == "source"
        )
        out = torch.empty_like(query)
        runtime.run(
            query,
            cu,
            lengths,
            role=role,
            pool=pool,
            out=out,
            bounds=[],
            query_starts=(),
            attention_workspace=decode_scratch(runtime),
        )
        pool = runtime.pool
        outputs.append(out)
    return outputs


def test_shared_decode_arena_matches_fresh_arenas(monkeypatch):
    from aleph_alpha_inference.kernels import serving_compressed
    from aleph_alpha_inference.kernels.serving_compressed import row_bounds

    compression = 16
    torch.manual_seed(1022)
    all_lengths = [65539, 33793, 70000, 4097, 52001, 98000, 129, 81920]
    cache, full_table = _cache(all_lengths)
    source, reindex, active = _impls(monkeypatch, compression, cache)
    live = torch.tensor(all_lengths, device="cuda", dtype=torch.int32)

    def batch(rows, shift):
        order = torch.arange(len(all_lengths), device="cuda").roll(shift)[:rows]
        q = torch.randn(rows, 1, 48, 128, device="cuda", dtype=torch.bfloat16)
        return (
            full_table[order].contiguous(),
            live[order].contiguous(),
            q,
            torch.randn_like(q),
        )

    bounds_calls = []
    monkeypatch.setattr(
        serving_compressed,
        "row_bounds",
        lambda *args: bounds_calls.append(1) or row_bounds(*args),
    )

    def check(rows, shift):
        table, lengths, q, reindex_q = batch(rows, shift)
        bounds_calls.clear()
        actual = _step(source, reindex, active, cache, table, lengths, q, reindex_q)
        # Both layers of the step share one row-bounds computation.
        assert len(bounds_calls) == 1
        expected = _reference(cache, table, lengths, q, reindex_q)
        for got, ref in zip(actual, expected):
            torch.testing.assert_close(got, ref, rtol=0, atol=0)

    # Interleave sizes so each runtime reuses arena rows another size overwrote.
    for rows, shift in ((3, 0), (1, 2), (5, 1), (3, 4), (8, 3), (2, 6)):
        check(rows, shift)

    # A graph captured at one size stays exact after eager sizes reuse its rows.
    table, lengths, q, reindex_q = batch(4, 5)
    _step(source, reindex, active, cache, table, lengths, q, reindex_q)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = _step(source, reindex, active, cache, table, lengths, q, reindex_q)
    for rows, shift in ((6, 2), (1, 7)):
        check(rows, shift)
    q.normal_()
    reindex_q.normal_()
    graph.replay()
    expected = _reference(cache, table, lengths, q, reindex_q)
    for got, ref in zip(outs, expected):
        torch.testing.assert_close(got, ref, rtol=0, atol=0)
    assert len({r.route.data_ptr() for r in source._decode_runtimes.values()}) == 1
