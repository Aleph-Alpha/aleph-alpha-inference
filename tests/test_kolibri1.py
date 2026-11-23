# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
import os
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from vllm import envs, LLM, SamplingParams
from vllm.config import KernelConfig
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.models.registry import ModelRegistry
from vllm.reasoning import ReasoningParserManager
from vllm.tool_parsers import ToolParserManager
from aleph_alpha_inference.config import Kolibri1Config
from vllm.model_executor.models.utils import PPMissingLayer
from aleph_alpha_inference import kolibri1
from aleph_alpha_inference.kolibri1 import (
    compute_layer_is_full_attention,
    expert_counts,
    Kolibri1ForCausalLM,
    pin_triton_for_small_fp8_blocks,
    sigmoid_logit_add_routing,
)
from aleph_alpha_inference.reasoning import Kolibri1ParserReasoningAdapter
from aleph_alpha_inference.tool_parser import Kolibri1ToolParser

from checkpoints import KOLIBRI1_CONFIG


def test_registration(register_plugin):
    assert "Kolibri1ForCausalLM" in ModelRegistry.get_supported_archs()


def test_config_registration(register_plugin):
    assert CONFIG_MAPPING[Kolibri1Config.model_type] is Kolibri1Config


def test_reasoning_parser_registration(register_plugin):
    for name in ("kolibri1", "kolibri"):
        parser_cls = ReasoningParserManager.get_reasoning_parser(name)
        assert parser_cls is Kolibri1ParserReasoningAdapter


def test_tool_parser_registration(register_plugin):
    for name in ("kolibri1", "kolibri"):
        assert ToolParserManager.get_tool_parser(name) is Kolibri1ToolParser


def test_config_parsing(register_plugin, kolibri1_model_dir):
    config = AutoConfig.from_pretrained(kolibri1_model_dir)
    assert isinstance(config, Kolibri1Config)
    assert config.model_type == Kolibri1Config.model_type
    assert config.architectures == ["Kolibri1ForCausalLM"]
    assert config.num_experts == 8
    assert config.head_dim == 32
    assert config.shared_expert_intermediate_size == 256
    assert config.sliding_window == 65
    assert config.rope_parameters["rope_theta"] == 10000.0
    assert config.norm_topk_prob is False

    # Survives the save/load round trip vLLM and HF both perform.
    reloaded = Kolibri1Config.from_dict(config.to_dict())
    assert reloaded.model_type == Kolibri1Config.model_type
    assert reloaded.sliding_window == config.sliding_window
    assert reloaded.layer_types == config.layer_types


def test_config_disables_deep_gemm(monkeypatch):
    # Given DeepGEMM and its E8M0 scales explicitly enabled
    monkeypatch.setenv("VLLM_USE_DEEP_GEMM", "1")
    monkeypatch.setenv("VLLM_USE_DEEP_GEMM_E8M0", "1")

    # When the config is built
    Kolibri1Config(**KOLIBRI1_CONFIG)

    # Then both are disabled
    assert os.environ["VLLM_USE_DEEP_GEMM"] == "0"
    assert os.environ["VLLM_USE_DEEP_GEMM_E8M0"] == "0"


def test_layer_pattern():
    """Hybrid attention pattern comes from layer_types; missing key errors."""
    config = Kolibri1Config(**KOLIBRI1_CONFIG)
    flags = compute_layer_is_full_attention(config)
    assert flags == [False, False, False, False, True, True]

    # layer_types is required: a config without it must fail loudly instead
    # of silently building some default attention pattern.
    config.layer_types = None
    with pytest.raises(ValueError):
        compute_layer_is_full_attention(config)


def _moe_vllm_config(num_experts: int, num_redundant_experts: int) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                num_experts=num_experts,
                vocab_size=16,
                hidden_size=8,
                tie_word_embeddings=False,
            )
        ),
        parallel_config=SimpleNamespace(
            eplb_config=SimpleNamespace(num_redundant_experts=num_redundant_experts)
        ),
        quant_config=None,
    )


def test_expert_counts_come_from_the_config():
    counts = expert_counts(_moe_vllm_config(64, 8), ep_size=4)
    assert counts == (64, 8, 72, 18)


def test_pipeline_rank_without_moe_blocks_builds(monkeypatch):
    """A pipeline-parallel rank that holds only dense-prefix layers
    (`mlp_only_layers`) has no MoE block to take the expert counts from."""
    dense_layer = SimpleNamespace(mlp=torch.nn.Identity())
    model = SimpleNamespace(
        layers=[dense_layer, PPMissingLayer(), PPMissingLayer()],
        make_empty_intermediate_tensors=None,
    )
    monkeypatch.setattr(kolibri1, "pin_triton_for_small_fp8_blocks", lambda _: None)
    monkeypatch.setattr(kolibri1, "Kolibri1Model", lambda **_: model)
    monkeypatch.setattr(kolibri1, "ParallelLMHead", lambda *_, **__: None)
    monkeypatch.setattr(kolibri1, "LogitsProcessor", lambda *_: None)
    monkeypatch.setattr(
        kolibri1,
        "get_ep_group",
        lambda: SimpleNamespace(device_group=SimpleNamespace(size=lambda: 4)),
    )

    rank = Kolibri1ForCausalLM(vllm_config=_moe_vllm_config(64, 8))

    assert rank.moe_layers == [] and rank.num_moe_layers == 0
    assert rank.num_logical_experts == rank.num_routed_experts == 64
    assert rank.num_redundant_experts == 8
    assert rank.num_physical_experts == 72
    assert rank.num_local_physical_experts == 18


@pytest.mark.parametrize(
    "block_size, linear_backend, expected",
    [
        ([64, 64], "auto", "triton"),
        ([128, 128], "auto", "auto"),
        ([64, 64], "cutlass", "cutlass"),
    ],
)
def test_small_fp8_blocks_pin_triton(block_size, linear_backend, expected):
    """Only auto selection of a non-128 FP8 block moves to Triton."""
    vllm_config = SimpleNamespace(
        quant_config=Fp8Config(
            is_checkpoint_fp8_serialized=True, weight_block_size=block_size
        ),
        kernel_config=KernelConfig(linear_backend=linear_backend),
    )

    pin_triton_for_small_fp8_blocks(vllm_config)

    assert vllm_config.kernel_config.linear_backend == expected


def test_routing_semantics():
    """sigmoid_logit_add routing must match torchtitan's router.

    Reference: torchtitan TokenChoiceTopKRouter with
    router_score_fn=sigmoid_logit_add, route_norm=False, route_scale=1.0;
    selection is top-k over (logits + expert_bias); the routing weights are
    the unbiased sigmoid(logits) of the selected experts.
    """
    torch.manual_seed(0)
    num_tokens, num_experts, top_k = 64, 384, 6
    logits = torch.randn(num_tokens, num_experts) * 3
    # Magnitudes as observed in a trained Kolibri 1 checkpoint (up to ~20).
    bias = torch.randn(num_experts) * 5
    hidden = torch.zeros(num_tokens, 8)  # unused by the routing function

    weights, ids = sigmoid_logit_add_routing(
        hidden_states=hidden,
        gating_output=logits.clone(),
        topk=top_k,
        renormalize=False,
        e_score_correction_bias=bias,
    )

    # Reference (torchtitan router.py).
    scores = torch.sigmoid(logits.float())
    scores_for_choice = logits.float() + bias
    ref_ids = torch.topk(scores_for_choice, k=top_k, dim=-1, sorted=True)[1]
    ref_weights = scores.gather(dim=-1, index=ref_ids)

    assert torch.equal(ids.long(), ref_ids), "expert selection mismatch"
    assert torch.allclose(weights, ref_weights), "routing weights mismatch"

    # The selection must differ from vLLM's built-in sigmoid scoring
    # (top-k over sigmoid(logits) + bias) whenever the bias is nonzero;
    # this is the reason the custom routing function exists.
    score_add_ids = torch.topk(scores + bias, k=top_k, dim=-1)[1]
    assert not torch.equal(
        ref_ids.sort(-1)[0], score_add_ids.sort(-1)[0]
    ), "test is vacuous: logit-add and score-add selected the same experts"


@pytest.mark.gpu
def test_model_loading(register_plugin, kolibri1_model_dir, tokenizer):
    llm = LLM(
        model=kolibri1_model_dir,
        tokenizer=tokenizer,
        enforce_eager=True,
        dtype="bfloat16",
        max_model_len=128,
        gpu_memory_utilization=0.3,
    )
    out = llm.generate(
        ["Hello world"],
        SamplingParams(max_tokens=8, temperature=0.0),
    )
    assert out[0].outputs[0].token_ids


@pytest.mark.gpu
def test_workers_run_without_deep_gemm(
    register_plugin, kolibri1_model_dir, tokenizer, monkeypatch
):
    # Given DeepGEMM and its E8M0 scales enabled where vLLM is launched
    monkeypatch.setenv("VLLM_USE_DEEP_GEMM", "1")
    monkeypatch.setenv("VLLM_USE_DEEP_GEMM_E8M0", "1")
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    # When vLLM starts its engine core and workers
    llm = LLM(
        model=kolibri1_model_dir,
        tokenizer=tokenizer,
        enforce_eager=True,
        dtype="bfloat16",
        max_model_len=128,
        gpu_memory_utilization=0.3,
        # A fixed KV cache skips vLLM's free-memory check, which fails while
        # the previous test's engine is still releasing GPU memory.
        kv_cache_memory_bytes=2**30,
    )

    # Then the workers see both disabled. The function is nested, so that
    # cloudpickle sends it by value: the workers cannot import this module.
    def deep_gemm_env(_worker) -> tuple[bool, bool]:
        return envs.VLLM_USE_DEEP_GEMM, envs.VLLM_USE_DEEP_GEMM_E8M0

    assert llm.collective_rpc(deep_gemm_env) == [(False, False)]


def test_indexed_attention_layers_attend_like_full_attention_layers():
    """Sparse layers named `indexed_attention` get no window and no RoPE."""
    config = Kolibri1Config(**KOLIBRI1_CONFIG)
    expected = compute_layer_is_full_attention(config)
    config.layer_types = [
        "indexed_attention" if kind == "full_attention" else kind
        for kind in config.layer_types
    ]

    assert compute_layer_is_full_attention(config) == expected
