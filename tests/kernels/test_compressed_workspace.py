# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""Routing scratch is bounded, isolated by lane, and profiled before KV sizing."""

from types import SimpleNamespace

import pytest
import torch

from sparse_hf_config import sparse_hf_config


@pytest.mark.gpu
def test_shared_scratch_retains_source_pool_lifetimes():
    from aleph_alpha_inference.kernels.serving_compressed import (
        routing_workspace,
        ServingCompressedLayer,
        source_pool,
    )

    device = torch.device("cpu")
    scratch = routing_workspace(device, (0, 0), 20, 32768, 2048, False)
    table = torch.zeros(2, 256, dtype=torch.int32)
    pools = [source_pool(device, (0, 0), source, 20) for source in ("a", "b")]
    runtimes = [
        ServingCompressedLayer(
            SimpleNamespace(),
            table,
            rows,
            span,
            2048,
            False,
            scratch=scratch,
            source_pool=pool,
        )
        for rows, span, pool in [(20, 16384, pools[0]), (8, 32768, pools[1])]
    ]
    assert runtimes[0].route.data_ptr() == runtimes[1].route.data_ptr()
    assert runtimes[0].pool.data_ptr() != runtimes[1].pool.data_ptr()
    runtimes[0].pool.fill_(7)
    runtimes[1].pool.fill_(11)
    assert (runtimes[0].pool == 7).all()
    assert runtimes[1].route.shape == (8, 2184)
    assert runtimes[1].block_scores.is_contiguous()
    assert routing_workspace(device, (0, 0), 20, 32768, 2048, False) is scratch
    other = routing_workspace(device, (0, 1), 20, 32768, 2048, False)
    assert other["route"].data_ptr() != scratch["route"].data_ptr()
    with pytest.raises(ValueError, match="capacity"):
        ServingCompressedLayer(
            SimpleNamespace(), table, 21, 32768, 2048, False, scratch=scratch
        )


@pytest.mark.parametrize("source", [False, True])
def test_compressed_profile_reserves_max_prefill_shape(monkeypatch, source):
    import aleph_alpha_inference.kernels.nvfp4_attention as nvfp4_attention
    import aleph_alpha_inference.kernels.prefill_staging as staging
    import aleph_alpha_inference.kernels.serving_compressed as compressed
    import aleph_alpha_inference.kernels.vllm_backend as backend

    config = SimpleNamespace(
        model_config=SimpleNamespace(
            max_model_len=266240,
            hf_text_config=sparse_hf_config(index_compression=16, index_topk=2048),
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=64, max_num_batched_tokens=32768),
        compilation_config=SimpleNamespace(max_cudagraph_capture_size=None),
        parallel_config=SimpleNamespace(use_ubatching=False),
    )
    from vllm.v1.worker.workspace import WorkspaceManager

    manager = WorkspaceManager(torch.device("cpu"), num_ubatches=1)
    monkeypatch.setattr(backend, "current_workspace_manager", lambda: manager)
    calls = []
    monkeypatch.setattr(backend, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(backend, "_workspace_owner", lambda: (0, 0))
    monkeypatch.setattr(
        compressed, "routing_workspace", lambda *args: calls.append(("scratch", args))
    )
    monkeypatch.setattr(
        compressed, "source_pool", lambda *args: calls.append(("pool", args))
    )
    # The profile run loads the NVFP4 extension, which needs nvcc.
    monkeypatch.setattr(nvfp4_attention, "native", lambda: None)
    monkeypatch.setattr(staging, "workspace", lambda *args: None)
    impl = backend.Block8Impl(48, 128, 128**-0.5, num_kv_heads=4)
    impl.sparse_role = "indexer" if source else "reindexer"
    impl._reserve_scratch(torch.device("cpu"))
    scratch = [args for name, args in calls if name == "scratch"]
    assert scratch == [
        (torch.device("cpu"), (0, 0), 32768, 524288, 2048, False, 16384),
        (torch.device("cpu"), (0, 0), 64, 266240, 2048, True, 16384),
    ]
    assert sum(name == "pool" for name, _ in calls) == 2 * int(source)


def test_prefill_runtime_serves_every_request_count(monkeypatch):
    import aleph_alpha_inference.kernels.serving_compressed as compressed
    import aleph_alpha_inference.kernels.vllm_backend as backend

    config = SimpleNamespace(
        model_config=SimpleNamespace(
            max_model_len=32768,
            hf_text_config=sparse_hf_config(index_compression=16, index_topk=2048),
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=64, max_num_batched_tokens=32768),
        compilation_config=SimpleNamespace(max_cudagraph_capture_size=None),
        parallel_config=SimpleNamespace(use_ubatching=False),
    )
    monkeypatch.setattr(backend, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(backend, "_workspace_owner", lambda: (0, 0))
    impl = backend.Block8Impl(48, 128, 128**-0.5, num_kv_heads=4)
    monkeypatch.setattr(
        impl, "_compressed_routing_workspace", lambda *_args: (None, None)
    )
    calls = []

    class Runtime:
        def __init__(self, cache, table, *args, **kwargs):
            self.pool = None

        def run(self, *args, **kwargs):
            calls.append((self, self.table.shape[0]))

    monkeypatch.setattr(compressed, "ServingCompressedLayer", Runtime)
    q = torch.empty(4, 1, 48, 128)
    cache = SimpleNamespace(storage=torch.empty(1))
    metadata = SimpleNamespace(
        num_actual_tokens=4,
        max_seq_len=32768,
        max_query_len=4,
        query_start_loc=None,
        seq_lens=None,
        row_bounds=[],
        query_starts=(),
        min_context=0,
    )
    for requests in (1, 2, 3):
        metadata.block_table = torch.empty(requests, 256, dtype=torch.int32)
        impl._forward_compressed(q, cache, metadata, "source", None, q)
    assert all(runtime is calls[0][0] for runtime, _ in calls)
    assert [rows for _, rows in calls] == [1, 2, 3]


def test_eager_decode_sizes_share_one_bounded_arena(monkeypatch):
    import aleph_alpha_inference.kernels.vllm_backend as backend
    from aleph_alpha_inference.kernels.serving_compressed import ServingCompressedLayer

    config = SimpleNamespace(
        model_config=SimpleNamespace(
            max_model_len=266240,
            hf_text_config=sparse_hf_config(index_compression=16, index_topk=2048),
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=96, max_num_batched_tokens=32768),
        parallel_config=SimpleNamespace(use_ubatching=False),
        compilation_config=SimpleNamespace(max_cudagraph_capture_size=128),
    )
    monkeypatch.setattr(backend, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(backend, "_workspace_owner", lambda: (0, 0))
    monkeypatch.setattr(backend, "current_workspace_manager", lambda: None)
    impl = backend.Block8Impl(48, 128, 128**-0.5, num_kv_heads=4)
    impl.sparse_role = "indexer"
    monkeypatch.setattr(impl, "_compressed_decode_workspace", lambda: None)
    runtimes = []

    def run(self, *args, **kwargs):
        runtimes.append(self)

    monkeypatch.setattr(ServingCompressedLayer, "run", run)
    cache = SimpleNamespace(storage=torch.empty(1))
    # Each distinct eager decode size once, then a size beyond max_num_seqs
    # that a captured decode graph may still use.
    sizes = list(range(1, 97)) + [128]
    for count in sizes:
        q = torch.empty(count, 1, 48, 128)
        metadata = SimpleNamespace(
            num_actual_tokens=count,
            max_seq_len=266240,
            max_query_len=1,
            query_start_loc=None,
            seq_lens=None,
            block_table=torch.empty(count, 2080, dtype=torch.int32),
            row_bounds=[],
            query_starts=(),
            min_context=0,
        )
        impl._forward_compressed(q, cache, metadata, "source", None, q)
    assert len(runtimes) == len(sizes)
    for name in ("qp", "route", "remote", "pool", "block_scores", "token_scores"):
        assert len({getattr(r, name).data_ptr() for r in runtimes}) == 1, name
    assert runtimes[-1].route.shape == (128, 2184)
    assert runtimes[-1].pool.shape == (128, 2048)
    assert len(impl._decode_runtimes) == 64
    with pytest.raises(ValueError, match="capacity"):
        q = torch.empty(129, 1, 48, 128)
        metadata.num_actual_tokens = 129
        metadata.block_table = torch.empty(129, 2080, dtype=torch.int32)
        impl._forward_compressed(q, cache, metadata, "source", None, q)
