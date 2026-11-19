# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""The production Kolibri 1 KV cache plan, resolved by vLLM's own planner.

Each block packs SWA pages in pairs, or the four global KV owners' pages
(K/V, routing keys, summaries and partial sums in one record each).
"""

import copy
import json

import pytest
import torch
from vllm.config import set_current_vllm_config
from vllm.engine.arg_utils import EngineArgs
from vllm.forward_context import set_forward_context
from vllm.v1.attention.backends import utils as backend_utils
from vllm.v1.attention.backends.utils import (
    get_supported_kv_cache_layouts,
    resolve_kv_cache_layout,
)
from vllm.v1.core.kv_cache_planning import (
    get_global_kv_cache_groups,
    get_kv_cache_configs,
)
from vllm.v1.core.kv_cache_utils import get_max_concurrency_for_kv_cache_config
from vllm.v1.kv_cache_interface import KVCacheLayout
from vllm.v1.worker.gpu.attn_utils import get_kv_cache_spec
from aleph_alpha_inference import register
from aleph_alpha_inference.kernels.vllm_backend import serving_backend
from aleph_alpha_inference.model.vllm_model import sparse_attention
from aleph_alpha_inference.model.vllm_replay import replay_layout
from aleph_alpha_inference.model.vllm_shared_kv import OP_NAME

# The tests import `vllm_swa` themselves: it needs vLLM's CUDA flash attention
# extensions, which the CPU-only CI runner lacks, so importing it here would
# fail collection there.


NUM_LAYERS = 50
GLOBAL = tuple(range(5, NUM_LAYERS + 1, 5))
KV_SOURCES = {5: 5, 10: 5, 15: 5, 20: 20, 25: 25, 30: 30, 35: 30, 40: 30, 45: 30, 50: 5}
AVAILABLE_MEMORY = 100 * 2**30
# K/V, routing keys, block-mean summaries (576 bytes padded to 1024) and
# partial sums: 84480 bytes per owner. The owners' four pages set the stride.
SWA_PAGE, OWNER_PAGE = 131072, 84480
BLOCK_STRIDE = 4 * OWNER_PAGE
NUM_BLOCKS = AVAILABLE_MEMORY // BLOCK_STRIDE


def model_config(max_model_len=266240):
    return {
        "architectures": ["Kolibri1_1ForCausalLM"],
        "model_type": "kolibri1",
        "hidden_size": 512,
        "num_hidden_layers": NUM_LAYERS,
        "num_attention_heads": 48,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "hidden_act": "silu",
        "max_position_embeddings": max_model_len,
        "rms_norm_eps": 1e-06,
        "vocab_size": 4096,
        "rope_theta": 10000.0,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 128,
        "shared_expert_intermediate_size": 128,
        "norm_topk_prob": False,
        "attention_bias": False,
        "tie_word_embeddings": False,
        "use_sliding_window": True,
        "sliding_window": 513,
        "layer_types": [
            "full_attention" if i in GLOBAL else "sliding_attention"
            for i in range(1, NUM_LAYERS + 1)
        ],
        "eos_token_id": 4095,
        "pad_token_id": 4095,
        # Indexers 5/15/20/30 and K/V owners 5/20/25/30 (one-based). The
        # config keys are zero-based.
        "index_layers": [4, 14, 19, 29],
        "kv_source_layers": [
            KV_SOURCES.get(layer, layer) - 1 for layer in range(1, NUM_LAYERS + 1)
        ],
        "index_topk": 2048,
        "index_pool_size": 16384,
        "index_compression": 16,
        "index_num_local_tokens": 128,
        "index_num_sink_tokens": 8,
        "index_num_heads": 12,
        "kv_cache_qat": True,
        "dtype": "bfloat16",
    }


def name(number):
    return f"model.layers.{number - 1}.self_attn.attn"


def attention_layers(config):
    from aleph_alpha_inference.model.vllm_swa import CuteSWAAttention, CuteSWABackend

    replay_tokens = replay_layout(config.model_config.hf_text_config)
    for number in range(1, NUM_LAYERS + 1):
        if number not in GLOBAL:
            layer = CuteSWAAttention(
                48,
                128,
                128**-0.5,
                num_kv_heads=4,
                cache_config=config.cache_config,
                per_layer_sliding_window=513,
                prefix=name(number),
                attn_backend=CuteSWABackend,
            )
            layer.replay_tokens = replay_tokens.prefix_replay_tokens
            continue
        sparse_attention(
            config, name(number), 48, 128, 4, kv_source_name=name(KV_SOURCES[number])
        )
    return (CuteSWABackend, serving_backend(16))


def engine_config(tmp_path, compilation_config, max_model_len=266240):
    register()
    (tmp_path / "config.json").write_text(json.dumps(model_config(max_model_len)))
    return EngineArgs(
        model=str(tmp_path),
        skip_tokenizer_init=True,
        kv_cache_dtype="bfloat16",
        max_model_len=max_model_len,
        max_num_batched_tokens=32768,
        enable_prefix_caching=False,
        compilation_config=compilation_config,
    ).create_engine_config()


def plan(tmp_path, monkeypatch, connector_layout=None, max_model_len=266240):
    monkeypatch.delenv("VLLM_KV_CACHE_LAYOUT", raising=False)
    monkeypatch.setattr(
        backend_utils, "get_kv_connector_cache_layout", lambda _: connector_layout
    )
    config = engine_config(
        tmp_path, {"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}, max_model_len
    )
    with set_current_vllm_config(config):
        backends = attention_layers(config)
        specs = get_kv_cache_spec(config)
        supported = [layout.name for layout in get_supported_kv_cache_layouts(backends)]
        layout = resolve_kv_cache_layout(config, [supported], specs.values())
        groups = get_global_kv_cache_groups(config, [specs])
        (planned,) = get_kv_cache_configs(config, [specs], [AVAILABLE_MEMORY], groups)
    return layout, specs, planned, config


OWNERS = sorted(set(KV_SOURCES.values()))


def expected_swa_groups():
    swa = [name(n) for n in range(1, NUM_LAYERS + 1) if n not in GLOBAL]
    return [swa[start::20] for start in range(20)]


@pytest.mark.gpu
@pytest.mark.parametrize("connector_layout", [None, "HND"])
def test_production_kv_cache_plan(tmp_path, monkeypatch, connector_layout):
    # A connector preferring a layer-outer layout (NIXL: HND) cannot unpack it.
    layout, specs, planned, _ = plan(tmp_path, monkeypatch, connector_layout)
    assert layout is KVCacheLayout.BLHNC
    assert {n: s.page_size_bytes for n, s in specs.items()} == {
        name(n): SWA_PAGE for n in range(1, NUM_LAYERS + 1) if n not in GLOBAL
    } | {name(n): OWNER_PAGE for n in OWNERS}
    assert planned.num_blocks == NUM_BLOCKS
    assert [g.layer_names for g in planned.kv_cache_groups] == expected_swa_groups() + [
        [name(n) for n in OWNERS]
    ]
    assert [
        (t.size, t.layers, t.layer_stride, t.block_stride, t.offset)
        for t in planned.kv_cache_tensors
    ] == [
        (NUM_BLOCKS * BLOCK_STRIDE, group, SWA_PAGE, BLOCK_STRIDE, 0)
        for group in expected_swa_groups()
    ] + [
        (
            NUM_BLOCKS * BLOCK_STRIDE,
            [name(n) for n in OWNERS],
            OWNER_PAGE,
            BLOCK_STRIDE,
            0,
        )
    ]
    # Every page of a block stays 512-byte aligned.
    assert all(t.offset % 512 == 0 for t in planned.kv_cache_tensors)


# Decode steps schedule one token per request; SWA groups then hold only their
# window, and long-context requests are bounded by the owners' blocks.
@pytest.mark.gpu
@pytest.mark.parametrize(
    "max_model_len,concurrency,decode_concurrency",
    [
        (16384, 60.32, 968.75),
        (65536, 15.12, 446.28),
        (266240, 13.97, 139.36),
        (1048576, 11.01, 37.86),
    ],
)
def test_production_kv_cache_concurrency(
    tmp_path, monkeypatch, max_model_len, concurrency, decode_concurrency
):
    *_, planned, config = plan(tmp_path, monkeypatch, max_model_len=max_model_len)
    with set_current_vllm_config(config):
        actual = get_max_concurrency_for_kv_cache_config(config, planned)
        config.scheduler_config = copy.copy(config.scheduler_config)
        config.scheduler_config.max_num_batched_tokens = 1
        decode = get_max_concurrency_for_kv_cache_config(config, planned)
    assert round(actual, 2) == concurrency
    assert round(decode, 2) == decode_concurrency


def shared_pair(config):
    with set_current_vllm_config(config):
        source = sparse_attention(config, name(30), 48, 128, 4, kv_source_name=name(30))
        consumer = sparse_attention(
            config, name(35), 48, 128, 4, kv_source_name=name(30)
        )
    return source, consumer


@pytest.mark.gpu
@pytest.mark.parametrize("mode", [0, 3])
def test_consumer_owns_no_cache_and_splits_compiled_graphs(tmp_path, mode):
    config = engine_config(
        tmp_path, {"mode": mode, "cudagraph_mode": "FULL_AND_PIECEWISE"}
    )
    _, consumer = shared_pair(config)
    assert consumer.get_kv_cache_spec(config) is None
    # Compiled piecewise graphs break at the consumer like at vLLM's attention.
    assert (OP_NAME in config.compilation_config.splitting_ops) == (mode == 3)


@pytest.mark.gpu
def test_consumer_attends_over_its_source_cache(tmp_path, monkeypatch):
    config = engine_config(tmp_path, {"mode": 0})
    source, consumer = shared_pair(config)
    source.kv_cache = torch.empty(3, 1, 128, 576, dtype=torch.uint8, device="cuda")
    calls = []

    def forward(layer, query, key, value, kv_cache, metadata, output):
        calls.append((layer, key, value, kv_cache, metadata))
        output.copy_(query)

    monkeypatch.setattr(consumer.impl, "forward", forward)
    metadata = object()
    query = torch.randn(5, 48 * 128, dtype=torch.bfloat16, device="cuda")
    with set_forward_context({name(30): metadata}, config):
        output = consumer(query, None, None)
    assert calls == [(consumer, None, None, source.kv_cache, metadata)]
    assert torch.equal(output, query)
