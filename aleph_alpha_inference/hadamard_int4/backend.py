# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""vLLM attention backend for the Hadamard int4 KV cache (Kolibri 1's full-attention
layers).

Registered under ``AttentionBackendEnum.CUSTOM`` and handed to those layers'
``Attention`` directly; every other layer keeps the backend vLLM selects for
``--kv-cache-dtype``.

Batches arrive decode-first. Decode rows (query length 1) run the CuTe int4 kernel,
which needs no host syncs and captures into full CUDA graphs (the builder declares
``UNIFORM_SINGLE_TOKEN_DECODE``). Prefill rows restore their cached history to bf16 and
run FlashAttention varlen; with chunked prefill the restored history is small next to
the chunk's attention FLOPs. Mixed batches run piecewise, the attention op outside the
graph.
"""

from dataclasses import dataclass, replace
from typing import ClassVar

import torch
from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import AttentionSpec

from aleph_alpha_inference.hadamard_int4 import (
    DECODE_TILE,
    HEAD_DIM,
    KV_DTYPE,
    STATE_BYTES,
    page,
    store,
)

# Upper bound on the transient bf16 K+V history one prefill chunk restores at once.
PREFILL_RESTORE_BYTES = 1 << 30


@dataclass
class HadamardInt4Metadata:
    num_actual_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    max_query_len: int
    max_seq_len: int
    query_start_loc: torch.Tensor
    query_start_loc_cpu: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor


class HadamardInt4MetadataBuilder(AttentionMetadataBuilder[HadamardInt4Metadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=False)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> HadamardInt4Metadata:
        meta = common_attn_metadata
        num_decodes, num_prefills, num_decode_tokens, _ = split_decodes_and_prefills(
            meta, decode_threshold=1
        )
        return HadamardInt4Metadata(
            num_actual_tokens=meta.num_actual_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            max_query_len=meta.max_query_len,
            max_seq_len=meta.max_seq_len,
            query_start_loc=meta.query_start_loc,
            query_start_loc_cpu=meta.query_start_loc_cpu,
            seq_lens=meta.seq_lens,
            block_table=meta.block_table_tensor,
            slot_mapping=meta.slot_mapping,
        )


class HadamardInt4Backend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [KV_DTYPE]
    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type["HadamardInt4Impl"]:
        return HadamardInt4Impl

    @staticmethod
    def get_builder_cls() -> type[HadamardInt4MetadataBuilder]:
        return HadamardInt4MetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(DECODE_TILE)]

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        return block_size is None or block_size % DECODE_TILE == 0

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [HEAD_DIM]

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        # Per (token, head): K codes 64 B + fp16 terms 4 B, then the same for V
        return replace(spec, state_content_bytes=STATE_BYTES)

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER


class HadamardInt4Impl(AttentionImpl[HadamardInt4Metadata]):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = KV_DTYPE,
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        **unsupported,
    ) -> None:
        if head_size != HEAD_DIM:
            raise ValueError(
                f"Hadamard int4 KV needs head_size {HEAD_DIM}, got {head_size}"
            )
        if abs(scale - head_size**-0.5) > 1e-6:
            raise ValueError(
                "Hadamard int4 KV supports only the 1/sqrt(head_size) softmax scale"
            )
        if sliding_window is not None or alibi_slopes is not None or logits_soft_cap:
            raise ValueError("Hadamard int4 KV serves causal full attention only")
        if kv_sharing_target_layer_name is not None:
            raise ValueError("Hadamard int4 KV does not support KV sharing")
        if any(v not in (None, False, -1) for v in unsupported.values()):
            raise ValueError(f"Hadamard int4 KV does not support {sorted(unsupported)}")
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads or num_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.attn_type = attn_type

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        tokens = slot_mapping.shape[0]
        store.pack(
            key[:tokens], value[:tokens], slot_mapping, page.cache_views(kv_cache)
        )

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: HadamardInt4Metadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("fused output quantization is not supported")
        if attn_metadata is None:  # profiling run
            return output.fill_(0)
        from aleph_alpha_inference.hadamard_int4 import decode as decode_kernel

        meta = attn_metadata
        cache = page.cache_views(kv_cache)
        decodes = meta.num_decodes
        if decodes:
            decode_kernel.decode(
                query[:decodes],
                cache,
                meta.block_table[:decodes],
                meta.seq_lens[:decodes],
                out=output[:decodes],
            )
        if meta.num_prefills:
            self._attend_prefills(query, output, cache, meta)
        return output

    def _attend_prefills(
        self, query, output, cache, meta: HadamardInt4Metadata
    ) -> None:
        from vllm.v1.attention.backends.fa_utils import get_flash_attn_version
        from vllm.vllm_flash_attn import flash_attn_varlen_func

        # Every size below is known on the host: no device syncs.
        max_len = -(-meta.max_seq_len // 16) * 16
        per_sequence = max_len * self.num_kv_heads * HEAD_DIM * 2 * query.element_size()
        group = max(1, PREFILL_RESTORE_BYTES // per_sequence)
        starts = meta.query_start_loc_cpu
        first = meta.num_decodes
        last = first + meta.num_prefills
        for begin in range(first, last, group):
            end = min(last, begin + group)
            # the restored histories packed back to back, sized by the host-side bound
            q0, q1 = int(starts[begin]), int(starts[end])
            history_lens = torch.zeros(
                end - begin + 1, device=query.device, dtype=torch.int32
            )
            torch.cumsum(meta.seq_lens[begin:end], 0, out=history_lens[1:])
            k, v = store.restore(
                cache, meta.block_table[begin:end], history_lens, max_len, query.dtype
            )
            flash_attn_varlen_func(
                q=query[q0:q1],
                k=k,
                v=v,
                max_seqlen_q=meta.max_query_len,
                cu_seqlens_q=meta.query_start_loc[begin : end + 1] - q0,
                max_seqlen_k=max_len,
                cu_seqlens_k=history_lens,
                softmax_scale=self.scale,
                causal=True,
                out=output[q0:q1],
                fa_version=get_flash_attn_version(),
            )
