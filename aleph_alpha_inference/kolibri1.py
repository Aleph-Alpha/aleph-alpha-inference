import os
from collections.abc import Iterable
from functools import partial
from itertools import islice
from typing import Any

import torch
from torch import nn

from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import FusedMoEFactory, GateLinear
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import (
    EagleModelMixin,
    MixtureOfExperts,
    SupportsEagle3,
    SupportsPP,
)
from vllm.model_executor.models.qwen3_moe import Qwen3MoeMLP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
    sequence_parallel_chunk,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backend import AttentionType


class Kolibri1Config(Qwen3MoeConfig):
    """HF config for Kolibri 1 checkpoints."""

    model_type = "kolibri1"

    # transformers generates `__init__` per config subclass; only `__post_init__` is inherited.
    def __post_init__(self, **kwargs: Any) -> None:
        super().__post_init__(**kwargs)
        # DeepGEMM would round our fp32 FP8 scales to powers of two
        os.environ["VLLM_USE_DEEP_GEMM"] = "0"
        os.environ["VLLM_USE_DEEP_GEMM_E8M0"] = "0"


def compute_layer_is_full_attention(config: Any) -> list[bool]:
    """Per-layer full-attention flags from the config's ``layer_types``.

    A layer is full attention when it is listed as ``"full_attention"``,
    sliding-window otherwise.
    """
    num_layers = config.num_hidden_layers
    layer_types = getattr(config, "layer_types", None)
    if layer_types is None:
        raise ValueError(
            "Kolibri 1 configs must define layer_types (one of "
            '"full_attention" / "sliding_attention" per layer).'
        )
    if len(layer_types) != num_layers:
        raise ValueError(
            f"layer_types has {len(layer_types)} entries but the model "
            f"has {num_layers} layers."
        )
    return [lt == "full_attention" for lt in layer_types]


class Kolibri1Attention(nn.Module):
    """Grouped-query attention with per-layer sliding-window and RNoPE."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str,
        is_full_attention: bool,
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size

        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        # Explicit head_dim: 128 at hidden_size 2560 with 48 heads, not
        # hidden_size // num_heads.
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        if is_full_attention:
            per_layer_sliding_window = None
        else:
            sliding_window = getattr(config, "sliding_window", None)
            if sliding_window is None or sliding_window <= 0:
                raise ValueError(
                    "Kolibri 1 configs must set sliding_window to a positive "
                    "token span for sliding-attention layers, got "
                    f"{sliding_window}."
                )
            per_layer_sliding_window = sliding_window

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        if is_full_attention:
            self.rotary_emb = None
        else:
            self.rotary_emb = get_rope(
                self.head_dim,
                max_position=config.max_position_embeddings,
                rope_parameters=config.rope_parameters,
                is_neox_style=True,
            )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=per_layer_sliding_window,
            attn_type=AttentionType.DECODER,
            prefix=f"{prefix}.attn",
        )

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q_by_head = q.view(
            *q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim
        )
        q = self.q_norm(q_by_head).view(q.shape)
        k_by_head = k.view(
            *k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim
        )
        k = self.k_norm(k_by_head).view(k.shape)

        if self.rotary_emb is not None:
            q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


@torch.compile(backend=current_platform.simple_compile_backend)
def sigmoid_logit_add_routing(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    e_score_correction_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """torchtitan's ``sigmoid_logit_add`` router score function."""
    logits = gating_output.float()
    scores_for_choice = logits + e_score_correction_bias
    topk_ids = torch.topk(scores_for_choice, k=topk, dim=-1)[1]
    topk_weights = torch.sigmoid(logits.gather(1, topk_ids))
    if renormalize:
        # torchtitan's route_norm: divide by the weight sum plus 1e-20.
        topk_weights = topk_weights / (
            topk_weights.sum(dim=-1, keepdim=True) + 1e-20
        )
    return topk_weights, topk_ids.to(torch.int32)


class Kolibri1SparseMoeBlock(nn.Module):
    """Sigmoid-logit-add-routed MoE with an internal router and one shared
    expert."""

    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        parallel_config = vllm_config.parallel_config
        quant_config = vllm_config.quant_config

        tp_size = get_tensor_model_parallel_world_size()
        ep_size = get_ep_group().device_group.size()
        self.n_routed_experts = config.num_experts

        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        if tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {tp_size} is greater than "
                f"the number of experts {config.num_experts}."
            )

        # Expert-count bookkeeping read by Kolibri1ForCausalLM's
        # MixtureOfExperts attributes and updated on elastic EP rescaling.
        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = parallel_config.eplb_config.num_redundant_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // ep_size

        # Router logits are computed in float32, matching torchtitan (the
        # trained expert bias has magnitudes up to ~20, so selection is
        # sensitive to logit precision).
        self.gate = GateLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            out_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )
        # torchtitan keeps the routing correction bias on the router in
        # float32; the checkpoint stores it as
        # ``model.layers.N.moe.router.expert_bias`` (rewritten to this
        # parameter by ``hf_to_vllm_mapper``).
        self.gate.e_score_correction_bias = nn.Parameter(
            torch.zeros(config.num_experts, dtype=torch.float32)
        )

        shared_expert_intermediate_size = getattr(
            config, "shared_expert_intermediate_size", None
        )
        if not shared_expert_intermediate_size:
            raise ValueError(
                "Kolibri 1 configs must set shared_expert_intermediate_size "
                "(Kolibri 1 always has one shared expert), got "
                f"{shared_expert_intermediate_size}."
            )
        # Kolibri 1 names the module ``shared_experts`` (plural), matching
        # the checkpoint layout.
        self.shared_experts = Qwen3MoeMLP(
            hidden_size=config.hidden_size,
            intermediate_size=shared_expert_intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            reduce_results=False,
            expert_gate=None,
            prefix=f"{prefix}.shared_experts",
        )

        self.experts = FusedMoEFactory(
            shared_experts=self.shared_experts,
            gate=self.gate,
            num_experts=self.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            enable_eplb=parallel_config.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=self.is_sequence_parallel,
            custom_routing_function=partial(
                sigmoid_logit_add_routing,
                e_score_correction_bias=self.gate.e_score_correction_bias,
            ),
            router_logits_dtype=self.gate.out_dtype,
        )
        # FusedMoEFactory builds an internal router when a gate is passed;
        # MoERunner has no is_internal_router flag, the router itself is the
        # equivalent invariant.
        assert self.experts.router is not None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert hidden_states.dim() == 2, (
            "Kolibri1SparseMoeBlock expects (num_tokens, hidden_size) inputs"
        )
        num_tokens = hidden_states.shape[0]

        if self.is_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)

        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=hidden_states
        )

        if self.is_sequence_parallel:
            final_hidden_states = tensor_model_parallel_all_gather(
                final_hidden_states, 0
            )
            final_hidden_states = final_hidden_states[:num_tokens]

        return final_hidden_states


class Kolibri1DecoderLayer(nn.Module):
    """Sandwich-norm decoder block: input/post-attn/ffn/post-ffn RMSNorms."""

    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config

        layer_idx = extract_layer_index(prefix)
        is_full_attention = compute_layer_is_full_attention(config)[layer_idx]

        self.self_attn = Kolibri1Attention(
            vllm_config=vllm_config,
            prefix=f"{prefix}.self_attn",
            is_full_attention=is_full_attention,
        )

        self.mlp = Kolibri1SparseMoeBlock(
            vllm_config=vllm_config, prefix=f"{prefix}.mlp"
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            positions=positions, hidden_states=hidden_states
        )
        hidden_states = self.post_attn_norm(hidden_states)

        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual
        )
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_ffn_norm(hidden_states)
        return hidden_states, residual


@support_torch_compile
class Kolibri1Model(nn.Module, EagleModelMixin):
    hf_to_vllm_mapper = WeightsMapper(
        # torchtitan keeps the router bias at ``moe.router.expert_bias``;
        # vLLM stores it on the gate as ``e_score_correction_bias``.
        orig_to_new_substr={
            ".moe.router.expert_bias": ".mlp.gate.e_score_correction_bias",
        },
        orig_to_new_stacked={
            ".q_proj": (".qkv_proj", "q"),
            ".k_proj": (".qkv_proj", "k"),
            ".v_proj": (".qkv_proj", "v"),
            # .experts.gate_up_proj is handled by MoERunner.load_weights.
            # Kolibri 1 names its shared expert ``shared_experts`` (plural).
            ".shared_experts.gate_proj": (".shared_experts.gate_up_proj", 0),
            ".shared_experts.up_proj": (".shared_experts.gate_up_proj", 1),
        },
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config

        self.config = config

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Kolibri1DecoderLayer(
                vllm_config=vllm_config, prefix=prefix
            ),
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size
            )
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        # EAGLE-3 aux hidden states: layer id k is the (pre-norm) output of
        # the k-th decoder layer, 0 being the embeddings, matching vLLM's
        # Qwen3-MoE convention. Empty unless set_aux_hidden_state_layers ran.
        aux_hidden_states = self._maybe_add_hidden_state(
            [], self.start_layer, hidden_states, residual
        )
        for layer_idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            hidden_states, residual = layer(positions, hidden_states, residual)
            self._maybe_add_hidden_state(
                aux_hidden_states, layer_idx + 1, hidden_states, residual
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            ignore_unexpected_suffixes=[
                ".bias",
                "_bias",
                ".weight_scale",
                "_weight_scale",
                ".input_scale",
                "_input_scale",
            ],
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


class Kolibri1ForCausalLM(nn.Module, SupportsPP, SupportsEagle3, MixtureOfExperts):
    hf_to_vllm_mapper = Kolibri1Model.hf_to_vllm_mapper
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        self.model = Kolibri1Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        # MixtureOfExperts bookkeeping (used for EPLB / expert parallel).
        # Every Kolibri 1 decoder layer carries an MoE block; with pipeline
        # parallelism only this rank's layers are real modules.
        local_moe_blocks = [
            layer.mlp
            for layer in self.model.layers
            if not isinstance(layer, PPMissingLayer)
        ]
        self.moe_layers = [moe.experts for moe in local_moe_blocks]
        example_layer = local_moe_blocks[0]

        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_shared_experts = 1
        self.num_logical_experts = example_layer.n_logical_experts
        self.num_physical_experts = example_layer.n_physical_experts
        self.num_local_physical_experts = example_layer.n_local_physical_experts
        self.num_routed_experts = example_layer.n_routed_experts
        self.num_redundant_experts = example_layer.n_redundant_experts

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            moe = layer.mlp
            moe.n_local_physical_experts = num_local_physical_experts
            moe.n_physical_experts = num_physical_experts
            moe.n_redundant_experts = self.num_redundant_experts
            moe.experts.update_expert_map()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
