# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""Serving defaults applied before vLLM selects workers and cache backends."""

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.config import VerifyAndUpdateConfig
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from aleph_alpha_inference.kernels.arch import sm80_port_enabled
from aleph_alpha_inference.kernels.serving_options import sparse_serving_options
from aleph_alpha_inference.kernels.sparse_layout import SparseCheckpointLayout
from aleph_alpha_inference.model.vllm_replay import bounded_replay_enabled, replay_layout

logger = init_logger(f"vllm.{__name__}")


class Kolibri1SparseConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_config(config: VllmConfig) -> None:
        layout = SparseCheckpointLayout.from_hf_config(
            config.model_config.hf_text_config
        )
        cache = config.cache_config
        attention = config.attention_config
        if cache.cache_dtype == "auto":
            cache.cache_dtype = "bfloat16"
        if not cache.user_specified_block_size:
            cache.block_size = 128
            # vLLM otherwise replaces this with FA4's BF16 default (16),
            # which is incompatible with the paged NVFP4 routing kernel.
            cache.user_specified_block_size = True
        if cache.block_size != 128:
            raise ValueError("Kolibri NVFP4 serving requires block_size=128")
        if attention.backend is None:
            attention.backend = AttentionBackendEnum.FLASH_ATTN
        flash_attn_version = 2 if sm80_port_enabled() else 4
        if attention.flash_attn_version is None:
            attention.flash_attn_version = flash_attn_version
        if (
            attention.backend != AttentionBackendEnum.FLASH_ATTN
            or attention.flash_attn_version != flash_attn_version
        ):
            raise ValueError(
                f"Kolibri sparse serving requires FLASH_ATTN with flash_attn_version={flash_attn_version}"
            )

        # A prefix hit still recomputes the bounded-replay window (20,480 tokens
        # for 40 SWA-513 layers), so caching pays off only for very long shared
        # prefixes and costs ~6% throughput on unrelated prompts. vLLM resolves
        # an omitted flag to enabled before this hook, hence the serving option.
        if not sparse_serving_options(config).prefix_caching:
            cache.enable_prefix_caching = False

        if bounded_replay_enabled(config):
            _configure_replay(config, layout)


def _configure_replay(config: VllmConfig, layout: SparseCheckpointLayout) -> None:
    parallel = config.parallel_config
    if not layout.share_kv:
        raise ValueError("Kolibri bounded replay requires shared KV")
    if parallel.pipeline_parallel_size != 1 or parallel.use_ubatching:
        raise ValueError("Kolibri bounded replay requires PP1 without microbatching")
    if config.speculative_config is not None:
        raise ValueError("Kolibri bounded replay does not support speculative decoding")
    if not config.use_v2_model_runner:
        raise ValueError("Kolibri bounded replay requires the V2 model runner")
    layout = replay_layout(config.model_config.hf_text_config)
    compilation = config.compilation_config
    if compilation.decoder_replay_trim_threshold <= layout.decoder_window:
        # vLLM's default suits DeepSeek's 128-token window.
        compilation.decoder_replay_trim_threshold = layout.decoder_window + 1
    logger.info(
        "Kolibri SWA bounded replay: decoder window %d, prefix replay %d tokens",
        layout.decoder_window,
        layout.prefix_replay_tokens,
    )
