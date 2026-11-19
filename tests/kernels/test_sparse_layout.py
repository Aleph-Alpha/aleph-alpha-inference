# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""`SparseCheckpointLayout` reads the sparse layout from the flat config keys."""

from pathlib import Path
from typing import Any

import pytest
from transformers import PretrainedConfig
from aleph_alpha_inference.kernels.sparse_layout import (
    SPARSE_LAYOUT_KEYS,
    SparseCheckpointLayout,
)
from aleph_alpha_inference.config import Kolibri1_1Config, Kolibri1Config

from sparse_hf_config import (
    exported_sparse_config,
    FULL_LAYERS,
    NUM_LAYERS,
    parse_hf_config,
)


def test_flat_keys_load_as_the_layout(register_plugin: None, tmp_path: Path) -> None:
    hf_config = parse_hf_config(exported_sparse_config(), tmp_path)
    assert hf_config.sliding_window == 513
    layout = SparseCheckpointLayout.from_hf_config(hf_config)
    assert layout.index_layers == (4, 14, 19, 29)
    assert layout.index_topk == 2048
    assert layout.index_pool_size == 16384
    assert layout.index_compression == 16
    assert layout.sparse_layers == FULL_LAYERS
    assert layout.share_kv is True
    assert layout.kv_producer_layers == (4, 19, 24, 29)
    assert [layout.kv_source_layers[index] for index in FULL_LAYERS] == [
        *(4, 4, 4, 19, 24, 29, 29, 29, 29, 4)
    ]


@pytest.mark.parametrize(
    "model_type, config_class",
    [
        ("kolibri1", Kolibri1Config),
        ("kolibri1_1", Kolibri1_1Config),
    ],
)
def test_every_sparse_model_type_loads_the_layout(
    register_plugin: None,
    tmp_path: Path,
    model_type: str,
    config_class: type[PretrainedConfig],
) -> None:
    config = exported_sparse_config(
        model_type=model_type, architectures=["Kolibri1_1ForCausalLM"]
    )
    hf_config = parse_hf_config(config, tmp_path)
    assert type(hf_config) is config_class
    assert hf_config.sliding_window == 513
    layout = SparseCheckpointLayout.from_hf_config(hf_config)
    assert layout.index_layers == (4, 14, 19, 29)


def test_without_shared_kv_every_sparse_layer_produces_its_own(
    register_plugin: None, tmp_path: Path
) -> None:
    config = exported_sparse_config(kv_source_layers=list(range(NUM_LAYERS)))
    layout = SparseCheckpointLayout.from_hf_config(parse_hf_config(config, tmp_path))
    assert layout.share_kv is False
    assert layout.kv_producer_layers == FULL_LAYERS


@pytest.mark.parametrize(
    "key, value",
    [
        ("index_num_heads", 16),
        ("index_num_local_tokens", 64),
        ("index_compression", 4),
        ("kv_source_layers", [0, 1, 2]),
        ("index_topk", "many"),
        ("kv_cache_qat", False),
    ],
)
def test_layouts_the_kernels_do_not_implement_are_rejected(
    register_plugin: None, tmp_path: Path, key: str, value: Any
) -> None:
    config = exported_sparse_config(**{key: value})
    with pytest.raises(ValueError, match=key):
        SparseCheckpointLayout.from_hf_config(parse_hf_config(config, tmp_path))


@pytest.mark.parametrize(
    "index_compression, index_num_sink_tokens", [(1, 8), (8, 4), (16, 4)]
)
def test_a_sink_count_the_kernels_do_not_use_is_rejected(
    register_plugin: None,
    tmp_path: Path,
    index_compression: int,
    index_num_sink_tokens: int,
) -> None:
    config = exported_sparse_config(
        index_compression=index_compression,
        index_num_sink_tokens=index_num_sink_tokens,
    )
    with pytest.raises(ValueError, match="index_num_sink_tokens"):
        SparseCheckpointLayout.from_hf_config(parse_hf_config(config, tmp_path))


@pytest.mark.parametrize("key", SPARSE_LAYOUT_KEYS)
def test_a_missing_key_is_named(
    register_plugin: None, tmp_path: Path, key: str
) -> None:
    config = exported_sparse_config()
    del config[key]
    with pytest.raises(ValueError, match=key):
        SparseCheckpointLayout.from_hf_config(parse_hf_config(config, tmp_path))


def test_indexed_attention_layers_are_sparse_layers(
    register_plugin: None, tmp_path: Path
) -> None:
    # Transformers names the sparse layers "indexed_attention"; earlier
    # checkpoints name them "full_attention". Both state the same layout.
    exported = exported_sparse_config()
    indexed = exported_sparse_config(
        layer_types=[
            "indexed_attention" if kind == "full_attention" else kind
            for kind in exported["layer_types"]
        ]
    )
    expected = SparseCheckpointLayout.from_hf_config(
        parse_hf_config(exported, tmp_path)
    )

    layout = SparseCheckpointLayout.from_hf_config(parse_hf_config(indexed, tmp_path))

    assert layout.sparse_layers == expected.sparse_layers == FULL_LAYERS
    assert layout.share_kv is expected.share_kv is True
    assert layout.kv_producer_layers == expected.kv_producer_layers
