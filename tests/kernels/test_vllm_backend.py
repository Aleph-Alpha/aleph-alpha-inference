# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""Cache accounting and metadata contracts for native paged serving."""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.kv_cache_interface import (
    create_kv_cache_views,
    FullAttentionSpec,
    KVCacheLayout,
    KVCacheTensor,
)
from aleph_alpha_inference.kernels.summary_cache import SummaryCache
from aleph_alpha_inference.kernels.vllm_backend import (
    Block8Backend,
    Block8MetadataBuilder,
    min_row_context,
    record_fields,
    serving_backend,
)

from sparse_hf_config import sparse_hf_config


@pytest.mark.parametrize("index_compression", [1, 8])
def test_serving_backend_rejects_other_routing_policies(index_compression):
    assert serving_backend(16) is Block8Backend
    with pytest.raises(ValueError, match="mean16"):
        serving_backend(index_compression)


def test_cache_spec_accounts_for_all_fields():
    page, bytes_per_token = 128, 660
    spec = Block8Backend.customize_spec(
        FullAttentionSpec(
            block_size=page,
            num_kv_heads=4,
            head_size=128,
            dtype=torch.bfloat16,
        )
    )
    payload = page * bytes_per_token
    assert spec.real_page_size_bytes == spec.page_size_bytes == payload
    assert payload % 512 == 0
    assert spec.get_block_stride_alignment() == 512
    backing = torch.empty(spec.page_size_bytes * 3, dtype=torch.uint8)
    descriptor = KVCacheTensor(
        size=backing.numel(),
        layers=["source"],
        layer_stride=backing.numel(),
        block_stride=spec.page_size_bytes,
    )
    (view,) = create_kv_cache_views(backing, spec, 3, KVCacheLayout.BLHNC, descriptor)
    assert view.shape == (3, 1, page, bytes_per_token)
    assert view.dtype == torch.uint8
    assert view.data_ptr() == backing.data_ptr()
    raw = view.view(3, -1)
    assert raw.stride(0) == spec.page_size_bytes
    assert raw[1].data_ptr() - raw[0].data_ptr() == spec.page_size_bytes


def test_record_fields_start_on_aligned_offsets():
    # K/V, routing keys, summaries (576 bytes padded to 1024), partial sums.
    fields = [(0, 73728), (73728, 9216), (82944, 576), (83968, 512)]
    assert record_fields(SummaryCache, 128) == fields
    assert all(start % 512 == 0 for start, _ in fields)
    assert list(SummaryCache.page_bytes(128)) == [width for _, width in fields]


@pytest.mark.parametrize("offset,stride", [(256, 4 * 84480), (0, 4 * 84480 + 256)])
def test_cache_view_rejects_unaligned_pages(monkeypatch, offset, stride):
    import aleph_alpha_inference.kernels.vllm_backend as backend

    config = SimpleNamespace(
        model_config=SimpleNamespace(
            max_model_len=32768,
            hf_text_config=sparse_hf_config(index_topk=2048),
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=1, max_num_batched_tokens=1),
        compilation_config=SimpleNamespace(max_cudagraph_capture_size=None),
        parallel_config=SimpleNamespace(use_ubatching=False),
    )
    monkeypatch.setattr(backend, "get_current_vllm_config", lambda: config)
    impl = backend.Block8Impl(48, 128, 128**-0.5, num_kv_heads=4)
    backing = torch.empty(4 * stride + 1024, dtype=torch.uint8)
    start = -backing.data_ptr() % 512 + offset
    pages = backing[start:].as_strided((3, 1, 128, 660), (stride, 0, 660, 1))
    layers = {"owner": SimpleNamespace(kv_cache=pages)}
    monkeypatch.setattr(
        backend,
        "get_forward_context",
        lambda: SimpleNamespace(no_compile_layers=layers),
    )
    with pytest.raises(ValueError, match="512-byte aligned"):
        impl._cache_view("owner")


def _metadata_inputs(query_length, speculative_config=None):
    """A builder and common metadata of three requests with `query_length` rows."""
    config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=1048576),
        speculative_config=speculative_config,
    )
    spec = FullAttentionSpec(
        block_size=128, num_kv_heads=4, head_size=128, dtype=torch.bfloat16
    )
    builder = Block8MetadataBuilder(spec, ["source"], config, torch.device("cpu"))
    seq_lens = torch.tensor([100, 1024, 2048], dtype=torch.int32)
    common = SimpleNamespace(
        num_actual_tokens=3 * query_length,
        max_query_len=query_length,
        max_seq_len=2048,
        query_start_loc=torch.arange(4, dtype=torch.int32) * query_length,
        query_start_loc_cpu=torch.arange(4, dtype=torch.int32) * query_length,
        seq_lens=seq_lens,
        seq_lens_cpu_upper_bound=seq_lens.clone(),
        block_table_tensor=torch.zeros(3, 8192, dtype=torch.int32),
        slot_mapping=torch.zeros(3 * query_length, dtype=torch.int64),
    )
    return builder, common


@pytest.mark.parametrize("query_length", [1, 7])
def test_metadata_preserves_request_table_without_query_expansion(query_length):
    builder, common = _metadata_inputs(query_length)
    metadata = builder.build(0, common)
    assert metadata.block_table is common.block_table_tensor
    assert metadata.query_start_loc is common.query_start_loc
    assert metadata.seq_lens is common.seq_lens
    assert metadata.max_seq_len == (1048576 if query_length == 1 else 2048)
    # The first row of the shortest request sees 100 - 7 + 1 tokens. Decode
    # steps leave the bound unknown.
    assert metadata.min_context == (0 if query_length == 1 else 94)
    replacement = torch.ones_like(common.block_table_tensor)
    changed = builder.update_block_table(metadata, replacement, common.slot_mapping)
    assert changed.block_table is replacement
    assert metadata.block_table is common.block_table_tensor
    assert changed.min_context == metadata.min_context


def test_full_cuda_graphs_hold_single_token_decodes_only():
    # Prefill metadata holds host values that a FULL CUDA graph would freeze.
    config = SimpleNamespace(model_config=SimpleNamespace(max_model_len=1048576))
    spec = FullAttentionSpec(
        block_size=128, num_kv_heads=4, head_size=128, dtype=torch.bfloat16
    )
    support = Block8MetadataBuilder.get_cudagraph_support(config, spec)
    assert support == AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE


def test_metadata_ignores_host_lengths_under_speculative_decoding():
    # Async speculative decoding overestimates host lengths of decode rows.
    builder, common = _metadata_inputs(7, speculative_config=SimpleNamespace())
    assert builder.build(0, common).min_context == 0


def test_min_row_context_skips_requests_without_query_rows():
    # The second and last requests are CUDA-graph padding: no query rows,
    # zero length.
    query_start_loc = torch.tensor([0, 300, 300, 364, 364], dtype=torch.int32)
    seq_lens = torch.tensor([16384 + 364, 0, 20000, 0], dtype=torch.int32)
    assert min_row_context(query_start_loc, seq_lens) == 16384 + 65


def test_min_row_context_is_zero_when_unknown():
    query_start_loc = torch.tensor([0, 4], dtype=torch.int32)
    assert min_row_context(query_start_loc, None) == 0
    empty = torch.tensor([0, 0], dtype=torch.int32)
    assert min_row_context(empty, torch.tensor([0], dtype=torch.int32)) == 0
