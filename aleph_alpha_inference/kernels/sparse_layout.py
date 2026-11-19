# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""Sparse attention layout of a checkpoint.

A sparse checkpoint states its layout as flat config keys (`index_layers`,
`kv_source_layers`, `index_topk`, ...), with zero-based layer indices.
`SparseCheckpointLayout` has one field per key.
"""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator
from transformers import PretrainedConfig

# The config keys that only a sparse checkpoint states.
SPARSE_LAYOUT_KEYS = (
    "index_layers",
    "kv_source_layers",
    "index_topk",
    "index_pool_size",
    "index_compression",
    "index_num_local_tokens",
    "index_num_sink_tokens",
    "index_num_heads",
    "kv_cache_qat",
)

# The `layer_types` entries of the layers that run sparse attention.
# Transformers names them "indexed_attention". Earlier Kolibri 1.1 checkpoints
# list them as "full_attention", like the full-attention layers of Kolibri 1.
SPARSE_LAYER_TYPES = ("indexed_attention", "full_attention")


class SparseCheckpointLayout(BaseModel):
    """Where and how a checkpoint runs sparse attention.

    Layer indices start at 0.

    Attributes:
        layer_types: Attention kind of every layer. The layers of a type in
            `SPARSE_LAYER_TYPES` run sparse attention.
        num_attention_heads: Number of query heads.
        num_key_value_heads: Number of K/V heads.
        index_layers: The indexers: the layers that select keys from the
            whole context. The other sparse layers select from the pool of
            the last indexer before them.
        kv_source_layers: For every layer, the layer whose K/V it reads.
        index_topk: Number of keys a query selects.
        index_pool_size: Number of keys an indexer keeps for the layers that
            select from its pool.
        index_compression: Number of tokens one index key covers.
        index_num_local_tokens: Number of most recent tokens every query
            attends to. The serving kernels fix it.
        index_num_sink_tokens: Number of leading tokens every query attends
            to. The serving kernels fix it: 4 without index compression, 8
            with it.
        index_num_heads: Number of query heads that score keys.
        kv_cache_qat: Whether the sparse layers were trained on NVFP4 K/V.
            The serving kernels always store NVFP4 K/V, so it must be true.
    """

    model_config = ConfigDict(frozen=True, from_attributes=True)

    layer_types: tuple[str, ...]
    num_attention_heads: int
    num_key_value_heads: int
    index_layers: tuple[int, ...]
    kv_source_layers: tuple[int, ...]
    index_topk: int
    index_pool_size: int
    index_compression: Literal[1, 8, 16]
    index_num_local_tokens: Literal[128]
    index_num_sink_tokens: int
    index_num_heads: int
    kv_cache_qat: Literal[True]

    @model_validator(mode="after")
    def _reject_inconsistent_keys(self) -> Self:
        """Reject keys that contradict each other."""
        num_index_heads = self.num_attention_heads // self.num_key_value_heads
        if self.index_num_heads != num_index_heads:
            raise ValueError(
                f"index_num_heads={self.index_num_heads}, but the indexers "
                f"score keys with the {num_index_heads} query heads of the first "
                "K/V head"
            )
        if len(self.kv_source_layers) != len(self.layer_types):
            raise ValueError(
                f"kv_source_layers has {len(self.kv_source_layers)} entries, "
                f"but the model has {len(self.layer_types)} layers"
            )
        num_sink_tokens = 4 if self.index_compression == 1 else 8
        if self.index_num_sink_tokens != num_sink_tokens:
            raise ValueError(
                f"index_num_sink_tokens={self.index_num_sink_tokens}, but the "
                f"serving kernels attend to {num_sink_tokens} sink tokens at "
                f"index_compression={self.index_compression}"
            )
        return self

    @classmethod
    def from_hf_config(cls, hf_config: PretrainedConfig) -> Self:
        """Read the sparse layout of a checkpoint.

        Args:
            hf_config: HF config of a sparse checkpoint.

        Returns:
            The layout.

        Raises:
            pydantic.ValidationError: A layout key is missing or mistyped, or
                the keys state a layout the serving kernels do not implement.
                This is a `ValueError`.
        """
        return cls.model_validate(hf_config)

    @property
    def sparse_layers(self) -> tuple[int, ...]:
        """The layers that run sparse attention."""
        return tuple(
            index
            for index, kind in enumerate(self.layer_types)
            if kind in SPARSE_LAYER_TYPES
        )

    @property
    def share_kv(self) -> bool:
        """Whether a sparse layer reads the K/V of another layer."""
        return any(
            self.kv_source_layers[index] != index for index in self.sparse_layers
        )

    @property
    def kv_producer_layers(self) -> tuple[int, ...]:
        """The layers whose K/V the sparse layers read, in ascending order."""
        return tuple(
            sorted({self.kv_source_layers[index] for index in self.sparse_layers})
        )
