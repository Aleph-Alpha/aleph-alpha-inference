# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""HF configs with a sparse layout for the tests."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from transformers import AutoConfig, PretrainedConfig

NUM_LAYERS = 50
# Zero-based indices of the full-attention layers 5, 10, ..., 50.
FULL_LAYERS = tuple(range(4, NUM_LAYERS, 5))


def sparse_hf_config(**overrides: Any) -> SimpleNamespace:
    """The sparse layout keys of a Kolibri 1 config without index compression
    and without shared K/V.

    Args:
        **overrides: Config entries that replace or extend the defaults.

    Returns:
        A stand-in for the HF config that `SparseCheckpointLayout.from_hf_config` reads.
    """
    config = {
        "layer_types": [
            "full_attention" if index in FULL_LAYERS else "sliding_attention"
            for index in range(NUM_LAYERS)
        ],
        "num_attention_heads": 48,
        "num_key_value_heads": 4,
        "index_layers": [4, 19, 29, 39],
        "kv_source_layers": list(range(NUM_LAYERS)),
        "index_topk": 512,
        "index_pool_size": 16384,
        "index_compression": 1,
        "index_num_local_tokens": 128,
        "index_num_heads": 12,
        "kv_cache_qat": True,
    } | overrides
    # The serving kernels fix the sink count by index compression.
    config.setdefault(
        "index_num_sink_tokens", 4 if config["index_compression"] == 1 else 8
    )
    return SimpleNamespace(**config)


def exported_sparse_config(**overrides: Any) -> dict[str, Any]:
    """The `config.json` of a sparse export with shared K/V and index
    compression, reduced to the keys the sparse layout depends on.

    Args:
        **overrides: Config entries that replace or extend the export.

    Returns:
        The config: indexers 5/15/20/30 and K/V owners 5/20/25/30
        (one-based), with layer 50 reading the K/V of layer 5.
    """
    kv_source_layers = list(range(NUM_LAYERS))
    kv_source_layers[9] = kv_source_layers[14] = kv_source_layers[49] = 4
    kv_source_layers[34] = kv_source_layers[39] = kv_source_layers[44] = 29
    return {
        "architectures": ["Kolibri1_1ForCausalLM"],
        "model_type": "kolibri1_1",
        "num_hidden_layers": NUM_LAYERS,
        "num_attention_heads": 48,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "use_sliding_window": True,
        "sliding_window": 513,
        "layer_types": [
            "full_attention" if index in FULL_LAYERS else "sliding_attention"
            for index in range(NUM_LAYERS)
        ],
        "index_layers": [4, 14, 19, 29],
        "kv_source_layers": kv_source_layers,
        "index_topk": 2048,
        "index_pool_size": 16384,
        "index_compression": 16,
        "index_num_local_tokens": 128,
        "index_num_sink_tokens": 8,
        "index_num_heads": 12,
        "kv_cache_qat": True,
    } | overrides


def parse_hf_config(config: dict[str, Any], tmp_path: Path) -> PretrainedConfig:
    """Parse `config` as vLLM parses the `config.json` of a checkpoint.

    Args:
        config: The content of `config.json`.
        tmp_path: An empty directory to write it to.

    Returns:
        The HF config. The plugin must be registered for its `model_type`.
    """
    (tmp_path / "config.json").write_text(json.dumps(config))
    return AutoConfig.from_pretrained(tmp_path)
