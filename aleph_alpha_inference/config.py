# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH

import os
from typing import TYPE_CHECKING

from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from vllm.model_executor.models.config import VerifyAndUpdateConfig

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig


class Kolibri1Config(Qwen3MoeConfig):
    model_type = "kolibri1"


class Kolibri1ForCausalLMConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_model_config(model_config: "ModelConfig") -> None:
        # Kolibri 1 FP8 weights carry fp32 block scales, which DeepGEMM would
        # round to powers of two (UE8M0).
        os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
        os.environ.setdefault("VLLM_USE_DEEP_GEMM_E8M0", "0")

    @staticmethod
    def verify_and_update_config(vllm_config: "VllmConfig") -> None:
        from aleph_alpha_inference import hadamard_int4

        if hadamard_int4.get_config(vllm_config) is None:
            return
        _size_kv_pool_for_hadamard_int4(vllm_config)


def _size_kv_pool_for_hadamard_int4(vllm_config: "VllmConfig") -> None:
    """Pick the shared KV block pool's block size for the int4 full-attention layers
    next to the sliding-window layers' ``--kv-cache-dtype``:
    the full layers take the block
    size whose page covers a minimal sliding block, and the sliding layers pad their
    pages up to it (vLLM honors ``skip_page_size_padded`` for any sliding-window
    spec)."""
    from vllm.utils.torch_utils import get_dtype_size, kv_cache_dtype_str_to_dtype

    from aleph_alpha_inference import hadamard_int4

    cache_config = vllm_config.cache_config
    model_config = vllm_config.model_config
    hf = model_config.hf_text_config
    if hf.head_dim != hadamard_int4.HEAD_DIM:
        raise ValueError(
            f"Hadamard int4 KV needs head_dim {hadamard_int4.HEAD_DIM}, "
            f"got {hf.head_dim}"
        )
    if cache_config.kv_cache_dtype_skip_layers:
        raise ValueError(
            "kolibri1.kv_cache=hadamard-int4 picks the full-attention layers' KV dtype "
            "itself; drop --kv-cache-dtype-skip-layers"
        )
    if cache_config.cache_dtype not in ("auto", "bfloat16", "fp8", "fp8_e4m3"):
        raise ValueError(
            "kolibri1.kv_cache=hadamard-int4 keeps --kv-cache-dtype for the "
            "sliding-window layers; use fp8 or auto (bf16), not "
            f"{cache_config.cache_dtype}"
        )
    parallel = vllm_config.parallel_config
    if (
        parallel.decode_context_parallel_size > 1
        or parallel.prefill_context_parallel_size > 1
    ):
        raise ValueError(
            "kolibri1.kv_cache=hadamard-int4 does not support context parallelism"
        )
    if vllm_config.speculative_config is not None:
        from vllm.logger import init_logger

        init_logger(__name__).warning(
            "kolibri1.kv_cache=hadamard-int4 with speculative decoding: verification "
            "steps take the prefill path (restore the history, then FlashAttention), "
            "not the int4 decode kernel"
        )
    sliding_dtype = kv_cache_dtype_str_to_dtype(cache_config.cache_dtype, model_config)
    sliding_bytes = 2 * hf.head_dim * get_dtype_size(sliding_dtype)
    requested = cache_config.block_size if cache_config.user_specified_block_size else 0
    block = hadamard_int4.full_attention_block_size(sliding_bytes, requested)
    num_kv_heads = model_config.get_num_kv_heads(vllm_config.parallel_config)
    cache_config.block_size = block
    cache_config.user_specified_block_size = True
    cache_config.skip_page_size_padded = (
        block * num_kv_heads * hadamard_int4.STATE_BYTES
    )
