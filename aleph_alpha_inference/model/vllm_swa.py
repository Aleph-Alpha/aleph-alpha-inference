# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""BF16 sliding-window attention (SWA) on FA4, for prefill, decode and cache
updates. Sparse attention keeps its 128-token pages.

Prefills honor SWA bounded replay: no query reads window KV below its request's
replay start (`replay_start`, see vllm_replay.py), which nothing has written.
"""

from dataclasses import dataclass, fields

import cutlass.cute as cute
import torch
from vllm.model_executor.layers.attention import Attention
from vllm.platforms import current_platform
from vllm.v1.attention.backends import fa_utils
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import is_quantized_kv_cache
from vllm.vllm_flash_attn.cute import interface as fa_interface, utils as fa_cute_utils

from aleph_alpha_inference.model.replay_cache import (
    clear_boundary,
    replay_cache_alignment,
)
from aleph_alpha_inference.model.vllm_replay import BoundedReplaySWASpec


# Batch size above which FA4 builds a batch lookup for its tile scheduler on
# sliding-window decode steps. FA4's own threshold is 256: above it, every
# call launches extra kernels to build the lookup. A decode step attends five
# key tiles per request in each of 40 layers, so these launches cost more than
# the lookup saves: on B200, FA4 decode fell 0.5 ms per step further behind
# from batch 256 to 512.
DECODE_BIN_BATCH_SEARCH_THRESH = 4096


@cute.jit
def replay_floor_mask(batch, head, query, key, seqlens, aux):
    return key >= aux[0][fa_cute_utils.ssa_to_scalar(batch)]


@dataclass
class CuteSWAMetadata(FlashAttentionMetadata):
    # Per request, the replay start relative to its rebased block table; None
    # when the batch holds no prefill.
    replay_floor: torch.Tensor | None = None


class CuteSWAMetadataBuilder(FlashAttentionMetadataBuilder):
    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.supports_draft_decode_metadata_update = False
        self.page = kv_cache_spec.block_size

    def build(
        self,
        common_prefix_len,
        common_attn_metadata,
        fast_build=False,
        replay_start=None,
    ):
        base = super().build(common_prefix_len, common_attn_metadata, fast_build)
        metadata = CuteSWAMetadata(**vars(base))
        if metadata.max_query_len == 1:
            # FA4 picks its KV split count from the context length, not from
            # the window: at decode batches of 16 or fewer on B200 it split
            # the window's five key tiles up to 32 ways and added a combine
            # launch per layer, about 0.5 ms per decode step over 40 layers.
            metadata.max_num_splits = 1
        # Decode rows sit above any replay start.
        elif replay_start is not None:
            self._rebase(metadata, replay_start)
        return metadata

    def _rebase(self, metadata: CuteSWAMetadata, replay_start: torch.Tensor) -> None:
        """Start each request's block table at the aligned tile holding its
        replay start, so the kernel loads no tile below it. FA4 SM100 D128 uses
        128-key tiles; keeping their absolute origin keeps FP8 probability
        rounding, which depends on the tile-local softmax scale."""
        alignment = replay_cache_alignment(self.page)
        origin_R = replay_start // alignment * alignment
        table = metadata.block_table
        columns = origin_R[:, None] // self.page + torch.arange(
            table.shape[1], device=table.device, dtype=origin_R.dtype
        )
        metadata.block_table = table.gather(
            1, columns.clamp_(max=table.shape[1] - 1).long()
        )
        metadata.seq_lens = metadata.seq_lens - origin_R
        metadata.replay_floor = replay_start - origin_R
        metadata.scheduler_metadata = None


class CuteSWABackend(FlashAttentionBackend):
    @classmethod
    def get_supported_kernel_block_sizes(cls, kv_cache_spec=None):
        return [64]

    @staticmethod
    def get_impl_cls():
        return CuteSWAImpl

    @staticmethod
    def get_builder_cls():
        return CuteSWAMetadataBuilder


class CuteSWAImpl(FlashAttentionImpl):
    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
        output_scale=None,
        output_block_scale=None,
    ):
        if getattr(attn_metadata, "replay_floor", None) is not None:
            return self._forward_replay(layer, query, kv_cache, attn_metadata, output)
        args = (
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale,
            output_block_scale,
        )
        # vLLM's profiling run passes no metadata.
        if attn_metadata is None or attn_metadata.max_query_len != 1:
            return super().forward(*args)
        # XXX: FA4 reads its threshold from a module global when called, and
        # takes no argument for it. Raise it for this call only, so prefill
        # and other models keep FA4's value. Remove when FA4 accepts the
        # threshold as an argument.
        default = fa_interface.BIN_BATCH_SEARCH_THRESH
        fa_interface.BIN_BATCH_SEARCH_THRESH = max(
            default, DECODE_BIN_BATCH_SEARCH_THRESH
        )
        try:
            return super().forward(*args)
        finally:
            fa_interface.BIN_BATCH_SEARCH_THRESH = default

    def _forward_replay(self, layer, query, kv_cache, metadata, output):
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        # The rebased tile below each replay start may hold another layout's
        # bytes; masked keys still multiply their values.
        clear_boundary(
            key_cache, value_cache, metadata.block_table, metadata.replay_floor
        )
        # Native Attention already quantized Q and populated the byte cache.
        # Match FA4's views and scale broadcasting after clearing raw bytes.
        if is_quantized_kv_cache(self.kv_cache_dtype):
            key_cache = key_cache.view(current_platform.fp8_dtype())
            value_cache = value_cache.view(current_platform.fp8_dtype())
        n = metadata.num_actual_tokens
        descale_shape = (metadata.query_start_loc.shape[0] - 1, self.num_kv_heads)
        fa_utils.flash_attn_varlen_func(
            q=query[:n],
            k=key_cache,
            v=value_cache,
            out=output[:n],
            cu_seqlens_q=metadata.query_start_loc,
            max_seqlen_q=metadata.max_query_len,
            seqused_k=metadata.seq_lens,
            max_seqlen_k=metadata.max_seq_len,
            softmax_scale=self.scale,
            causal=True,
            window_size=self.sliding_window,
            block_table=metadata.block_table,
            fa_version=4,
            num_splits=metadata.max_num_splits,
            q_descale=(
                layer._q_scale.expand(descale_shape)
                if self.supports_quant_query_input
                else None
            ),
            k_descale=layer._k_scale.expand(descale_shape),
            v_descale=layer._v_scale.expand(descale_shape),
            mask_mod=replay_floor_mask,
            aux_tensors=[metadata.replay_floor],
        )
        return output


class CuteSWAAttention(Attention):
    """Sliding-window attention whose KV is rebuilt by bounded replay when
    `replay_tokens` is set (see vllm_replay.py)."""

    replay_tokens = 0

    def get_kv_cache_spec(self, vllm_config):
        spec = super().get_kv_cache_spec(vllm_config)
        if not self.replay_tokens:
            return spec
        return BoundedReplaySWASpec(
            **{f.name: getattr(spec, f.name) for f in fields(spec) if f.init},
            replay_tokens=self.replay_tokens,
        )


def install_swa_decode(layers, vllm_config, replay_tokens=0):
    from vllm.model_executor.models.utils import PPMissingLayer

    for layer in layers:
        if isinstance(layer, PPMissingLayer):
            continue
        original = layer.self_attn.attn
        if original.sliding_window is None:
            continue
        name = original.layer_name
        del vllm_config.compilation_config.static_forward_context[name]
        attn = CuteSWAAttention(
            original.num_heads,
            original.head_size,
            original.impl.scale,
            num_kv_heads=original.num_kv_heads,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            per_layer_sliding_window=original.sliding_window,
            prefix=name,
            attn_backend=CuteSWABackend,
        )
        attn.replay_tokens = replay_tokens
        layer.self_attn.attn = attn
