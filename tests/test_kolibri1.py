# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH

import os

import pytest
import torch


def test_registration(register_plugin):
    from vllm.model_executor.models.registry import ModelRegistry

    assert "Kolibri1ForCausalLM" in ModelRegistry.get_supported_archs()


def test_config_registration(register_plugin):
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    from aleph_alpha_inference.config import Kolibri1Config

    assert CONFIG_MAPPING[Kolibri1Config.model_type] is Kolibri1Config


def test_reasoning_parser_registration(register_plugin):
    from vllm.reasoning import ReasoningParserManager

    from aleph_alpha_inference.reasoning import Kolibri1ParserReasoningAdapter

    parser_cls = ReasoningParserManager.get_reasoning_parser("kolibri1")
    assert parser_cls is Kolibri1ParserReasoningAdapter


def test_tool_parser_registration(register_plugin):
    from vllm.tool_parsers import ToolParserManager
    from vllm.tool_parsers.hermes_tool_parser import Hermes2ProToolParser

    assert ToolParserManager.get_tool_parser("kolibri1") is Hermes2ProToolParser


def test_config_parsing(register_plugin, kolibri1_model_dir):
    from transformers import AutoConfig

    from aleph_alpha_inference.config import Kolibri1Config

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


DEEP_GEMM_ENV = ("VLLM_USE_DEEP_GEMM", "VLLM_USE_DEEP_GEMM_E8M0")


@pytest.fixture
def unset_deep_gemm_env(monkeypatch):
    # setenv first, so monkeypatch also undoes what the config hook sets.
    for name in DEEP_GEMM_ENV:
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)


def _run_config_hook() -> None:
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP

    MODELS_CONFIG_MAP["Kolibri1ForCausalLM"].verify_and_update_model_config(None)


def test_config_hook_disables_deep_gemm_by_default(
    register_plugin, unset_deep_gemm_env
):
    # Given DeepGEMM left unset where vLLM is launched
    # When vLLM builds the Kolibri 1 ModelConfig
    _run_config_hook()

    # Then both default to disabled
    assert [os.environ[name] for name in DEEP_GEMM_ENV] == ["0", "0"]


def test_config_hook_keeps_explicit_deep_gemm(
    register_plugin, unset_deep_gemm_env, monkeypatch
):
    # Given DeepGEMM and its E8M0 scales explicitly enabled
    for name in DEEP_GEMM_ENV:
        monkeypatch.setenv(name, "1")

    # When vLLM builds the Kolibri 1 ModelConfig
    _run_config_hook()

    # Then the explicit setting wins
    assert [os.environ[name] for name in DEEP_GEMM_ENV] == ["1", "1"]


def test_routing_semantics():
    """sigmoid_logit_add routing must match torchtitan's router.

    Reference: torchtitan TokenChoiceTopKRouter with
    router_score_fn=sigmoid_logit_add, route_norm=False, route_scale=1.0;
    selection is top-k over (logits + expert_bias); the routing weights are
    the unbiased sigmoid(logits) of the selected experts.
    """
    from aleph_alpha_inference.kolibri1 import sigmoid_logit_add_routing

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
    assert not torch.equal(ref_ids.sort(-1)[0], score_add_ids.sort(-1)[0]), (
        "test is vacuous: logit-add and score-add selected the same experts"
    )


@pytest.mark.gpu
def test_model_loading(register_plugin, kolibri1_model_dir, tokenizer):
    from vllm import LLM, SamplingParams

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


def _deep_gemm_env(_worker) -> tuple[bool, bool]:
    from vllm import envs

    return envs.VLLM_USE_DEEP_GEMM, envs.VLLM_USE_DEEP_GEMM_E8M0


@pytest.mark.gpu
def test_workers_run_without_deep_gemm(
    register_plugin, kolibri1_model_dir, tokenizer, unset_deep_gemm_env, monkeypatch
):
    from vllm import LLM

    # Given DeepGEMM left unset where vLLM is launched
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

    # Then the workers see both disabled
    assert llm.collective_rpc(_deep_gemm_env) == [(False, False)]
