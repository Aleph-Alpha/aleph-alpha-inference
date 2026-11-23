# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""Sparse attention extension of the Kolibri 1 model in `kolibri1.py`.

The native plugin retains checkpoint loading, FP8 linears/MoE, TP/DP, norms,
SWA, tokenization and logits. Only attention modules are replaced, and
encoder-decoder layouts run vLLM's decoder SWA bounded replay (vllm_replay.py).

Under torch.compile (VLLM_COMPILE) replay splits the model into two compiled
regions, like Gemma's YOCO fast prefill: the encoder runs up to the first
decoder layer's KV write, and the decoder runs on the whole batch or, in eager
steps that trim, on the replay batch.
"""

from typing import Any, cast

import torch
from torch import nn
from vllm.compilation.backends import set_model_tag
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CompilationMode
from vllm.forward_context import in_piecewise_cudagraph
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention.attention import unified_kv_cache_update
from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
from vllm.model_executor.layers.quantization.online.fp8 import (
    Fp8PerBlockOnlineLinearMethod,
)
from vllm.model_executor.models.utils import PPMissingLayer
from vllm.models.deepseek_v41.decoder_replay_layers import DecoderReplayLayers
from vllm.utils.torch_utils import _encode_layer_name

from aleph_alpha_inference.kernels.sparse_layout import SparseCheckpointLayout
from aleph_alpha_inference.kernels.vllm_backend import serving_backend
from aleph_alpha_inference.kolibri1 import Kolibri1ForCausalLM
from aleph_alpha_inference.model.vllm_replay import bounded_replay_enabled, replay_layout
from aleph_alpha_inference.model.vllm_shared_kv import SharedKVAttention

logger = init_logger(f"vllm.{__name__}")


class QueryWeightView:
    """Live Q-only FP8 weight views without registration or duplicate storage."""

    weight_scale = None
    input_scale = None
    input_scale_ub = None

    def __init__(self, projection, num_query_rows):
        self.projection = projection
        self.num_query_rows = num_query_rows

    @property
    def weight(self):
        return self.projection.weight[: self.num_query_rows]

    @property
    def weight_scale_inv(self):
        block_rows = self.projection.weight_block_size[0]
        return self.projection.weight_scale_inv[: self.num_query_rows // block_rows]


def sparse_attention(
    vllm_config, name, num_heads, head_dim, num_kv_heads, *, kv_source_name
):
    """Sparse attention `name` over the KV cache of layer `kv_source_name`."""
    layout = SparseCheckpointLayout.from_hf_config(
        vllm_config.model_config.hf_text_config
    )
    args = (num_heads, head_dim, head_dim**-0.5)
    kwargs = {
        "num_kv_heads": num_kv_heads,
        "cache_config": vllm_config.cache_config,
        "quant_config": vllm_config.quant_config,
        "prefix": name,
        "attn_backend": serving_backend(layout.index_compression),
    }
    if kv_source_name != name:
        return SharedKVAttention(*args, kv_source_name=kv_source_name, **kwargs)
    return Attention(*args, **kwargs)


def metadata_layer_name(attention):
    """The layer whose attention metadata `attention` reads."""
    if isinstance(attention, SharedKVAttention):
        return attention.kv_source_name
    return attention.layer_name


class NativeSparseAttention(nn.Module):
    qkv_proj: Any
    o_proj: Any
    q_norm: Any
    k_norm: Any
    q_size: int
    kv_size: int
    head_dim: int
    num_heads: int
    num_kv_heads: int

    def __init__(
        self,
        original,
        *,
        vllm_config,
        source_name,
        role,
        share_kv,
        kv_source_name=None,
    ):
        super().__init__()
        for name in ("qkv_proj", "o_proj", "q_norm", "k_norm"):
            setattr(self, name, getattr(original, name))
        for name in ("q_size", "kv_size", "head_dim", "num_heads", "num_kv_heads"):
            setattr(self, name, getattr(original, name))
        self.share_kv = share_kv and (
            kv_source_name != original.attn.layer_name
            if kv_source_name is not None
            else role == "reindexer"
        )
        if self.share_kv:
            self.k_norm = None
        self._query_weight_view = QueryWeightView(self.qkv_proj, self.q_size)
        name = original.attn.layer_name
        context = vllm_config.compilation_config.static_forward_context
        del context[name]
        self.attn = sparse_attention(
            vllm_config,
            name,
            self.num_heads,
            self.head_dim,
            self.num_kv_heads,
            kv_source_name=(kv_source_name or source_name) if self.share_kv else name,
        )
        self.attn.impl.sparse_role = role
        self.attn.impl.source_attention = context[source_name]
        # Set on the first decoder replay layer, whose KV the replay layers'
        # `write_batch_kv` writes for every row before they run on their batch.
        self.decoder_replay_layers: DecoderReplayLayers | None = None

    def _query_projection(self, hidden_states):
        method = self.qkv_proj.quant_method
        if isinstance(method, (Fp8LinearMethod, Fp8PerBlockOnlineLinearMethod)):
            if isinstance(method, Fp8LinearMethod) and (
                not method.block_quant or method.use_marlin
            ):
                raise ValueError(
                    "Shared-KV FP8 currently requires native block-FP8 weights"
                )
            if self.q_size % self.qkv_proj.weight_block_size[0]:
                raise ValueError("Query projection must align with FP8 output blocks")
            return method.apply(self._query_weight_view, hidden_states)
        if self.qkv_proj.weight.dtype in (torch.bfloat16, torch.float16):
            return torch.nn.functional.linear(
                hidden_states, self.qkv_proj.weight[: self.q_size]
            )
        raise ValueError("Unsupported query-only projection quantization")

    def _qkv(self, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim))
        return q, k, v.view(-1, self.num_kv_heads, self.head_dim)

    def forward(self, positions, hidden_states):
        if self.share_kv:
            q = self._query_projection(hidden_states)
            k = v = None
        else:
            q, k, v = self._qkv(hidden_states)
            replay = self.decoder_replay_layers
            if replay is not None and replay.replay_batch is not None:
                # `write_kv` already stored every row's KV; the cache summaries
                # accumulate, so the replay batch's rows must not append again.
                k = v = None
        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).flatten(1)
        output = self.attn(q, k, v)
        return self.o_proj(output)[0]

    def write_kv(self, hidden_states):
        """Store the KV `forward` would, without attending."""
        _, k, v = self._qkv(hidden_states)
        unified_kv_cache_update(k, v, self.attn.layer_name)


@support_torch_compile
class ReplayEncoder(nn.Module):
    """Embedding, the encoder layers and the first decoder layer up to its KV
    write: returns that layer's normed query, the residual and the write's
    dependency token (an output, so the write is not dead code)."""

    def __init__(self, *, vllm_config, prefix="", model, decoder_start):
        super().__init__()
        self.embed_tokens = model.embed_tokens
        self.layers = model.layers[: decoder_start + 1]

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ):
        hidden_states = (
            self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        )
        residual = None
        for layer in self.layers[:-1]:
            hidden_states, residual = layer(positions, hidden_states, residual)
        first = self.layers[-1]
        attention = first.self_attn
        x, residual = first.input_layernorm(hidden_states, residual)
        q, k, v = attention._qkv(x)
        q = attention.q_norm(q.view(-1, attention.num_heads, attention.head_dim))
        written = torch.ops.vllm.unified_kv_cache_update(
            k, v, _encode_layer_name(attention.attn.layer_name)
        )
        return q.flatten(1), residual, written


@support_torch_compile
class ReplayDecoder(nn.Module):
    """The decoder layers from the first one's attention, on the rows given,
    and the final norm."""

    def __init__(self, *, vllm_config, prefix="", model, decoder_start):
        super().__init__()
        self.layers = model.layers[decoder_start:]
        self.norm = model.norm

    def forward(
        self, query: torch.Tensor, residual: torch.Tensor, positions: torch.Tensor
    ):
        # Kolibri1DecoderLayer.forward past the attention input; the encoder
        # wrote this layer's KV.
        first = self.layers[0]
        attention = first.self_attn
        hidden_states = attention.o_proj(attention.attn(query, None, None))[0]
        hidden_states = first.post_attn_norm(hidden_states)
        hidden_states, residual = first.post_attention_layernorm(
            hidden_states, residual
        )
        hidden_states = first.post_ffn_norm(first.mlp(hidden_states))
        for layer in self.layers[1:]:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


def _require_tp1(parallel_config):
    if parallel_config.tensor_parallel_size != 1:
        raise ValueError("FP4 block8 currently requires TP1; use native DP/EP or PP")


def _require_supported_gpu() -> None:
    """Reject a GPU the sparse kernels are not built for: SM100 (e.g. B200)
    and SM90 (H100) serve. Reads the current CUDA device's compute capability.
    """
    # Other GPUs would fail later with an opaque CUDA error.
    major, minor = torch.cuda.get_device_capability()
    if major != 10 and (major, minor) != (9, 0):
        raise ValueError(
            "Kolibri sparse serving requires an SM100 GPU (compute capability "
            f"10.x, e.g. B200) or SM90 (H100); got {major}.{minor}"
        )


# The class name follows the model name Kolibri 1.1.
class Kolibri1_1ForCausalLM(Kolibri1ForCausalLM):  # noqa: N801
    model: Any
    # vLLM's DeepSeek-V4.1 model state reads these (see vllm_model_state.py).
    token_lookback_depth = 0
    decoder_replay_layers: DecoderReplayLayers | None = None

    @staticmethod
    def get_model_state_cls():
        from aleph_alpha_inference.model.vllm_model_state import Kolibri1ModelState

        return Kolibri1ModelState

    def load_weights(self, weights):
        # Borrowed keys are normalized by their owner, so neither case needs a
        # local parameter. Leave every other key to the native strict loader.
        unused_norms = {
            f"{name}.k_norm.weight"
            for name, module in cast(nn.Module, self).named_modules()
            if isinstance(module, NativeSparseAttention) and module.share_kv
        }
        return super().load_weights(
            (name, value) for name, value in weights if name not in unused_norms
        )

    def __init__(self, *, vllm_config, prefix=""):
        _require_supported_gpu()
        hf_config = vllm_config.model_config.hf_text_config
        if (
            hf_config.num_attention_heads,
            hf_config.num_key_value_heads,
            hf_config.head_dim,
        ) != (48, 4, 128):
            raise ValueError("Kolibri sparse serving requires native Q48/KV4/D128")
        if vllm_config.parallel_config.decode_context_parallel_size != 1:
            raise NotImplementedError(
                "Sparse routing does not yet support decode context parallelism"
            )
        sparse_layout = SparseCheckpointLayout.from_hf_config(hf_config)
        _require_tp1(vllm_config.parallel_config)
        indexers = sparse_layout.index_layers
        sparse_layers = set(sparse_layout.sparse_layers)
        if (
            not indexers
            or tuple(sorted(set(indexers))) != indexers
            or not set(indexers) <= sparse_layers
            or min(sparse_layers) != indexers[0]
        ):
            raise ValueError(
                "Sparse inference requires index_layers in ascending order that "
                "start at the first sparse layer"
            )
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        from aleph_alpha_inference.model.vllm_swa import install_swa_decode

        layout = (
            replay_layout(hf_config) if bounded_replay_enabled(vllm_config) else None
        )
        install_swa_decode(
            self.model.layers,
            vllm_config,
            replay_tokens=layout.prefix_replay_tokens if layout else 0,
        )
        source_name = None
        owner_names = {}
        for index, layer in enumerate(self.model.layers):
            if isinstance(layer, PPMissingLayer) or index not in sparse_layers:
                continue
            role = "indexer" if index in indexers else "reindexer"
            if role == "indexer":
                source_name = layer.self_attn.attn.layer_name
            elif source_name is None:
                raise ValueError(
                    "Each PP stage must start with an indexer before its reindexers; "
                    "set VLLM_PP_LAYER_PARTITION for the exported indexer layout"
                )
            owner_names[index] = layer.self_attn.attn.layer_name
            owner = sparse_layout.kv_source_layers[index]
            if owner not in owner_names:
                raise ValueError("KV owner must precede its consumer in this PP stage")
            layer.self_attn = NativeSparseAttention(
                layer.self_attn,
                vllm_config=vllm_config,
                source_name=source_name,
                role=role,
                share_kv=sparse_layout.share_kv,
                kv_source_name=owner_names[owner],
            )

        if layout is not None:
            self._install_decoder_replay(vllm_config, layout)

    def _install_decoder_replay(self, vllm_config, layout):
        mode = vllm_config.compilation_config.mode
        if mode not in (CompilationMode.NONE, CompilationMode.VLLM_COMPILE):
            logger.warning_once(
                "Kolibri decoder SWA bounded replay needs compilation mode 0 or 3; "
                "it is off, prefix replay stays on."
            )
            return
        self.decoder_replay_start = layout.decoder_start
        replay_layers = self.model.layers[layout.decoder_start :]
        if replay_layers[0].self_attn.share_kv:
            raise ValueError("The first decoder replay layer must own its KV")
        compiled = mode == CompilationMode.VLLM_COMPILE
        self.decoder_replay_layers = DecoderReplayLayers(
            layout.decoder_window,
            self._run_compiled_decoder if compiled else self._run_replay_layers,
            # Compiled, the encoder writes the first decoder layer's KV always.
            (lambda *_: None) if compiled else self._write_replay_kv,
            {metadata_layer_name(layer.self_attn.attn) for layer in replay_layers},
            replay_layers[0].self_attn.attn.layer_name,
        )
        if compiled:
            self._install_compiled_regions(vllm_config, layout.decoder_start)
        else:
            first = replay_layers[0].self_attn
            first.decoder_replay_layers = self.decoder_replay_layers
        logger.info_once(
            "Kolibri decoder SWA bounded replay (%s): in prefill steps without "
            "CUDA graphs, layer %d writes its KV for every token, and the rest "
            "of it and layers %d-%d run on each request's last %d tokens only. "
            "Prefix hits replay %d tokens.",
            "compiled" if compiled else "eager",
            layout.decoder_start,
            layout.decoder_start + 1,
            len(self.model.layers) - 1,
            layout.decoder_window,
            layout.prefix_replay_tokens,
        )

    def _install_compiled_regions(self, vllm_config, decoder_start):
        # Kolibri1Model's compiled forward never runs; the regions replace it.
        self.model.do_not_compile = True
        with set_model_tag("kolibri_replay_encoder"):
            self.replay_encoder = ReplayEncoder(
                vllm_config=vllm_config,
                model=self.model,
                decoder_start=decoder_start,
            )
        with set_model_tag("kolibri_replay_decoder"):
            self.replay_decoder = ReplayDecoder(
                vllm_config=vllm_config,
                model=self.model,
                decoder_start=decoder_start,
            )
        # A PIECEWISE graph's inputs keep their capture addresses, and the
        # encoder's last graph releases its outputs to the graph pool: the
        # decoder reads copies. PIECEWISE batches are at most this size.
        rows = vllm_config.compilation_config.max_cudagraph_capture_size or 0
        first = self.model.layers[decoder_start].self_attn
        hidden_size = vllm_config.model_config.hf_text_config.hidden_size
        options = {
            "dtype": vllm_config.model_config.dtype,
            # Online-quantized weights sit on the meta device until loading.
            "device": vllm_config.device_config.device,
        }
        self._decoder_inputs = (
            torch.empty(rows, first.q_size, **options),
            torch.empty(rows, hidden_size, **options),
        )

    def _run_compiled_decoder(self, query, residual, positions):
        if in_piecewise_cudagraph():
            num_tokens = query.shape[0]
            query_buffer, residual_buffer = self._decoder_inputs
            query = query_buffer[:num_tokens].copy_(query)
            residual = residual_buffer[:num_tokens].copy_(residual)
        return (self.replay_decoder(query, residual, positions),)

    def _write_replay_kv(self, hidden_states, residual, positions):
        layer = self.model.layers[self.decoder_replay_start]
        # The fused residual norm updates its inputs in place; the replay
        # batch still reads them.
        x, _ = layer.input_layernorm(hidden_states.clone(), residual.clone())
        layer.self_attn.write_kv(x)

    def _run_replay_layers(self, hidden_states, residual, positions):
        for layer in self.model.layers[self.decoder_replay_start :]:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.model.norm(hidden_states, residual)
        return (hidden_states,)

    def forward(
        self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None
    ):
        if self.decoder_replay_layers is None:
            return super().forward(
                input_ids, positions, intermediate_tensors, inputs_embeds
            )
        if hasattr(self, "replay_decoder"):
            query, residual, _ = self.replay_encoder(
                input_ids, positions, inputs_embeds
            )
            (hidden_states,) = self.decoder_replay_layers(query, residual, positions)
            return hidden_states
        # Kolibri1Model.forward with the decoder on the replay batch; replay
        # requires one pipeline stage.
        model = self.model
        hidden_states = (
            model.embed_input_ids(input_ids) if inputs_embeds is None else inputs_embeds
        )
        residual = None
        for layer in model.layers[: self.decoder_replay_start]:
            hidden_states, residual = layer(positions, hidden_states, residual)
        (hidden_states,) = self.decoder_replay_layers(
            hidden_states, residual, positions
        )
        return hidden_states
