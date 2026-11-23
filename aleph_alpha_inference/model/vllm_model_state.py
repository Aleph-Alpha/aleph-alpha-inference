# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""vLLM's DeepSeek-V4.1 model state, adapted to Kolibri's layers and backends.

`DeepseekV41ModelState` implements SWA bounded replay model-agnostically except
for two DeepSeek details this module replaces: which metadata builders take the
replay starts, and the input buffers of the decoder replay CUDA graphs.
"""

import json
from copy import copy

import torch
from vllm.compilation.breakable_cudagraph import (
    BreakableCUDAGraphWrapper,
    is_breakable_cudagraph_enabled,
)
from vllm.config import CompilationMode, CUDAGraphMode
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.models.deepseek_v41.nvidia.decoder_replay_cudagraph import (
    DecoderReplayCudaGraphManager,
)
from vllm.models.deepseek_v41.nvidia.model_state import (
    DeepseekV41ModelState,
    ReplayAttnMetadata,
)
from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
from vllm.v1.worker.gpu.model_states.default import DefaultModelState

from aleph_alpha_inference.model.cache_audit import audit_cache
from aleph_alpha_inference.model.vllm_swa import CuteSWAMetadataBuilder
from aleph_alpha_inference.model.vllm_warmup import warm_up_attention

logger = init_logger(f"vllm.{__name__}")


class Kolibri1ReplayAttnMetadata(ReplayAttnMetadata):
    """Hands the batch's replay starts to Kolibri's sliding-window builders."""

    def get_extra_attn_kwargs(self, attn_metadata_builder, num_reqs):
        if isinstance(attn_metadata_builder, CuteSWAMetadataBuilder):
            return {"replay_start": self.replay_start[:num_reqs]}
        return {}


class _ReplayStarts(DefaultModelState):
    """Sits between `DeepseekV41ModelState` and `DefaultModelState` in the MRO,
    so every replay start the DeepSeek state builds reaches Kolibri builders."""

    def prepare_attn(self, *args, model_specific_attn_metadata=None, **kwargs):
        # XXX: DeepseekV41ModelState names DeepSeek's SWA builder in
        # ReplayAttnMetadata. Remove when vLLM lets models choose the builders.
        if type(model_specific_attn_metadata) is ReplayAttnMetadata:
            model_specific_attn_metadata = Kolibri1ReplayAttnMetadata(
                model_specific_attn_metadata.replay_start
            )
        return super().prepare_attn(
            *args, model_specific_attn_metadata=model_specific_attn_metadata, **kwargs
        )


class Kolibri1DecoderReplayCudaGraphManager(DecoderReplayCudaGraphManager):
    """`DecoderReplayCudaGraphManager` with the inputs of Kolibri's decoder:
    hidden states, residual and positions."""

    def __init__(self, vllm_config, device, layers):
        # XXX: Mirrors DecoderReplayCudaGraphManager.__init__, which sizes its
        # buffers for DeepSeek's hyper-connection inputs. Remove when vLLM
        # lets models declare the replay inputs.
        compilation = copy(vllm_config.compilation_config)
        scheduler = vllm_config.scheduler_config
        sizes = compilation.decoder_replay_cudagraph_capture_sizes
        if not sizes:
            bound = min(
                compilation.max_cudagraph_capture_size,
                scheduler.max_num_seqs * layers.window,
            )
            sizes, size = [bound], layers.window
            while size < bound:
                sizes.append(size)
                size += max(layers.window, 1 << max(size.bit_length() - 4, 0))
        sizes = sorted(set(sizes))
        if max(sizes) > scheduler.max_num_batched_tokens:
            raise ValueError(
                "decoder_replay_cudagraph_capture_sizes must not exceed max_num_batched_tokens"
            )
        compilation.cudagraph_capture_sizes = sizes
        compilation.max_cudagraph_capture_size = max(sizes)
        replay_config = copy(vllm_config)
        replay_config.compilation_config = compilation
        CudaGraphManager.__init__(
            self, replay_config, device, CUDAGraphMode.PIECEWISE, 1
        )
        self.layers = layers
        self.breakable_cg_runner = BreakableCUDAGraphWrapper(
            layers.run_layers, replay_config
        )
        hidden_size = vllm_config.model_config.hf_text_config.hidden_size
        dtype = vllm_config.model_config.dtype
        num_tokens = compilation.max_cudagraph_capture_size
        self.input_buffers = [
            torch.zeros(num_tokens, hidden_size, dtype=dtype, device=device),
            torch.zeros(num_tokens, hidden_size, dtype=dtype, device=device),
            torch.zeros(num_tokens, dtype=torch.int64, device=device),
        ]


class Kolibri1ModelState(DeepseekV41ModelState, _ReplayStarts):
    """DeepSeek-V4.1's bounded replay for Kolibri; logs the KV cache audit and
    compiles Kolibri's prefill attention executables before serving."""

    _attention_warmup_pending = False

    def __init__(self, vllm_config, model, encoder_cache, device):
        compilation = vllm_config.compilation_config
        replay_graphs = compilation.cudagraph_decoder_replay
        # XXX: Keep DeepseekV41ModelState from building DeepSeek's replay
        # graphs; the Kolibri ones follow. Remove with the manager's XXX.
        compilation.cudagraph_decoder_replay = False
        try:
            super().__init__(vllm_config, model, encoder_cache, device)
        finally:
            compilation.cudagraph_decoder_replay = replay_graphs
        layers = self.decoder_replay_layers
        if (
            layers is not None
            and replay_graphs
            # Compiled, the decoder region runs replay batches in no graph.
            and compilation.mode == CompilationMode.NONE
            and compilation.cudagraph_mode.has_piecewise_cudagraphs()
            and is_breakable_cudagraph_enabled()
            and get_pp_group().is_first_rank
            # The graphs' fixed inputs hold no LoRA mapping and no None input ids.
            and vllm_config.lora_config is None
            and not vllm_config.model_config.enable_prompt_embeds
        ):
            threshold = compilation.decoder_replay_trim_threshold
            if threshold <= layers.window:
                raise ValueError(
                    "decoder_replay_trim_threshold must exceed the replay window "
                    f"of {layers.window} tokens: smaller graphs have nothing to trim"
                )
            self.replay_cudagraphs = Kolibri1DecoderReplayCudaGraphManager(
                vllm_config, device, layers
            )
            layers.trim_threshold = threshold

    def initialize_kv_cache(self, kv_cache_config, block_tables):
        super().initialize_kv_cache(kv_cache_config, block_tables)
        audit = audit_cache(self.vllm_config.compilation_config.static_forward_context)
        audit["planned_bytes"] = sum(t.size for t in kv_cache_config.kv_cache_tensors)
        logger.info(
            "Kolibri physical KV cache audit: %s", json.dumps(audit, sort_keys=True)
        )
        # vLLM binds a small cache to profile memory, then the sized one. The
        # profiling warmup compiles and loads every executable before KV sizing
        # counts their memory; executables do not depend on the cache size, so
        # the second warmup finds nothing to compile.
        self._attention_warmup_pending = True

    def capture_inner_cudagraphs(
        self, input_buffers, block_tables, attn_groups, kv_cache_config
    ):
        # Capture is the last startup step that runs before vLLM's JIT monitor
        # and receives the attention groups, so prefill executables compile here.
        if self._attention_warmup_pending:
            self._attention_warmup_pending = False

            def prepare(batch, tables, slot_mappings):
                replay = None
                if self._replay is not None and self._replay[0]:
                    replay = ReplayAttnMetadata(
                        self._replay_start[: batch.num_reqs].zero_()
                    )
                return _ReplayStarts.prepare_attn(
                    self,
                    batch,
                    CUDAGraphMode.NONE,
                    tables,
                    slot_mappings,
                    attn_groups,
                    kv_cache_config,
                    model_specific_attn_metadata=replay,
                )

            warm_up_attention(
                self, prepare, input_buffers, block_tables, kv_cache_config
            )
        super().capture_inner_cudagraphs(
            input_buffers, block_tables, attn_groups, kv_cache_config
        )
