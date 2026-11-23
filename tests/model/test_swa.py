# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""Native cache and dispatch contracts for sliding-window attention."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import SlidingWindowSpec
from aleph_alpha_inference.model.vllm_replay import BoundedReplaySWASpec

# vllm_swa imports vLLM's CuTe flash-attention, which raises at import without
# vLLM's CUDA extensions (e.g. on the CPU-only CI runner). So every test needs
# CUDA, and each imports vllm_swa itself, after the CPU run deselected it.
pytestmark = pytest.mark.gpu


def test_swa_page_selection_is_independent_of_sparse():
    from vllm.model_executor.layers.attention.attention import (
        _largest_kernel_block_within,
    )
    from aleph_alpha_inference.model.vllm_swa import CuteSWABackend

    # Unshared budget for block_size=128, then the tightest page that fits 64.
    assert _largest_kernel_block_within(CuteSWABackend, 2048, 2048 * 128, 128) == 64
    assert _largest_kernel_block_within(CuteSWABackend, 2048, 2048 * 64, 128) == 64
    assert CuteSWABackend.forward_includes_kv_cache_update is False


@pytest.mark.parametrize("query_len", [1, 2, None])
def test_dispatch_is_native_and_decode_raises_batch_search_threshold(
    monkeypatch, query_len
):
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
    from vllm.vllm_flash_attn.cute import interface as fa_interface
    from aleph_alpha_inference.model.vllm_swa import (
        CuteSWAImpl,
        DECODE_BIN_BATCH_SEARCH_THRESH,
    )

    impl = object.__new__(CuteSWAImpl)
    output = torch.empty(2)
    seen = []
    default = fa_interface.BIN_BATCH_SEARCH_THRESH
    assert default < DECODE_BIN_BATCH_SEARCH_THRESH

    def native(self, *args):
        seen.append((args, fa_interface.BIN_BATCH_SEARCH_THRESH))
        return output

    monkeypatch.setattr(FlashAttentionImpl, "forward", native)
    # vLLM's profiling run passes no metadata.
    metadata = (
        None
        if query_len is None
        else SimpleNamespace(replay_floor=None, max_query_len=query_len)
    )
    assert impl.forward(None, None, None, None, None, metadata, output) is output
    args, threshold = seen[0]
    assert args[5] is metadata
    # Only decode steps raise FA4's threshold, and only during the call.
    assert threshold == (DECODE_BIN_BATCH_SEARCH_THRESH if query_len == 1 else default)
    assert fa_interface.BIN_BATCH_SEARCH_THRESH == default


@pytest.mark.parametrize("query_len,num_splits", [(1, 1), (2, 0)])
def test_decode_metadata_disables_kv_splits(query_len, num_splits):
    """Decode steps run FA4 without KV splits; prefill keeps FA4's heuristic (0)."""
    from aleph_alpha_inference.model.vllm_swa import (
        CuteSWAMetadata,
        CuteSWAMetadataBuilder,
    )

    config = VllmConfig()
    config.model_config = SimpleNamespace(
        get_num_attention_heads=lambda _: 48,
        hf_config=SimpleNamespace(model_type="kolibri1"),
        hf_text_config=SimpleNamespace(),
        rswa_window=None,
        is_mm_prefix_lm=False,
    )
    spec = SlidingWindowSpec(
        block_size=64,
        num_kv_heads=4,
        head_size=128,
        dtype=torch.bfloat16,
        sliding_window=513,
    )
    builder = CuteSWAMetadataBuilder(spec, [], config, torch.device("cuda"))
    batch = 4
    common = SimpleNamespace(
        num_reqs=batch,
        num_actual_tokens=batch * query_len,
        max_query_len=query_len,
        max_seq_len=262144,
        query_start_loc=torch.arange(batch + 1, device="cuda", dtype=torch.int32)
        * query_len,
        seq_lens=torch.tensor([2, 513, 576, 262144], device="cuda", dtype=torch.int32),
        block_table_tensor=torch.zeros(batch, 4096, device="cuda", dtype=torch.int32),
        slot_mapping=torch.arange(batch * query_len, device="cuda"),
        causal=True,
        mm_req_doc_ranges=None,
        rswa_prefix_lens=None,
    )
    metadata = builder.build(0, common)
    assert isinstance(metadata, CuteSWAMetadata)
    assert metadata.max_num_splits == num_splits


def test_install_uses_native_attention_registration_and_cache_spec():
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.layers.attention import Attention
    from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
    from aleph_alpha_inference.model.vllm_swa import CuteSWAImpl, install_swa_decode

    config = VllmConfig()
    config.cache_config.block_size = 128
    config.cache_config.cache_dtype = "bfloat16"
    config.attention_config.flash_attn_version = 4
    config.model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(),
        dtype=torch.bfloat16,
        is_mm_prefix_lm=False,
    )
    with set_current_vllm_config(config):
        original = Attention(
            48,
            128,
            128**-0.5,
            num_kv_heads=4,
            cache_config=config.cache_config,
            per_layer_sliding_window=513,
            prefix="swa",
            attn_backend=FlashAttentionBackend,
        )
        full = Attention(
            48,
            128,
            128**-0.5,
            num_kv_heads=4,
            cache_config=config.cache_config,
            prefix="full",
            attn_backend=FlashAttentionBackend,
        )
        layers = [
            SimpleNamespace(self_attn=SimpleNamespace(attn=x)) for x in (original, full)
        ]
        install_swa_decode(layers, config, replay_tokens=20480)
        installed = layers[0].self_attn.attn
        assert isinstance(installed.impl, CuteSWAImpl)
        assert installed.impl.scale == original.impl.scale
        assert config.compilation_config.static_forward_context["swa"] is installed
        spec = installed.get_kv_cache_spec(config)
        assert isinstance(spec, BoundedReplaySWASpec)
        assert spec.block_size == 64 and spec.sliding_window == 513
        assert not spec.prefix_cacheable and spec.prefix_replay_tokens == 20480
        installed.replay_tokens = 0
        assert type(installed.get_kv_cache_spec(config)) is SlidingWindowSpec
        assert layers[1].self_attn.attn is full
        assert full.get_kv_cache_spec(config).block_size == 128
