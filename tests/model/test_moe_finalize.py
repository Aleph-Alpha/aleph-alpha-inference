# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""The plugin makes vLLM call FlashInfer's MoE with the unfused finalize."""

from types import ModuleType

import pytest
import torch
import vllm
from vllm.model_executor.layers.fused_moe.experts import flashinfer_cutlass_moe
from aleph_alpha_inference.model.vllm_moe_finalize import (
    install_deterministic_moe_finalize,
)

from tensor_asserts import assert_equal

# vLLM's FlashInfer MoE module and the kwargs of each call to its stand-in.
FlashInferModule = tuple[ModuleType, list[dict[str, object]]]


@pytest.fixture
def flashinfer_module(monkeypatch: pytest.MonkeyPatch) -> FlashInferModule:
    """vLLM's FlashInfer MoE module, with a stand-in that records its kwargs."""
    calls: list[dict[str, object]] = []

    def fake(*args: object, **kwargs: object) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(flashinfer_cutlass_moe, "flashinfer_cutlass_fused_moe", fake)
    return flashinfer_cutlass_moe, calls


def test_passes_unfused_finalize(flashinfer_module: FlashInferModule) -> None:
    module, calls = flashinfer_module
    assert install_deterministic_moe_finalize("1")
    module.flashinfer_cutlass_fused_moe(input=None)
    assert calls == [{"input": None, "use_fused_finalize": False}]


def test_installing_twice_wraps_once(flashinfer_module: FlashInferModule) -> None:
    module, _ = flashinfer_module
    assert install_deterministic_moe_finalize("1")
    wrapped = module.flashinfer_cutlass_fused_moe
    assert install_deterministic_moe_finalize("1")
    assert module.flashinfer_cutlass_fused_moe is wrapped


def test_mode_0_keeps_vllm_call(flashinfer_module: FlashInferModule) -> None:
    module, calls = flashinfer_module
    assert not install_deterministic_moe_finalize("0")
    module.flashinfer_cutlass_fused_moe(input=None)
    assert calls == [{"input": None}]


def test_other_vllm_version_is_skipped(
    flashinfer_module: FlashInferModule, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, calls = flashinfer_module
    monkeypatch.setattr(vllm, "__version__", "0.34.0")
    assert not install_deterministic_moe_finalize("1")
    module.flashinfer_cutlass_fused_moe(input=None)
    assert calls == [{"input": None}]


def test_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="KOLIBRI_VLLM_DETERMINISTIC_MOE"):
        install_deterministic_moe_finalize("check")


@pytest.mark.gpu
def test_real_kernel_repeats_bit_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    """FlashInfer's real kernel, called as the plugin makes vLLM call it, gives
    the same output on every call."""
    # Restores vLLM's function after the test.
    monkeypatch.setattr(
        flashinfer_cutlass_moe,
        "flashinfer_cutlass_fused_moe",
        flashinfer_cutlass_moe.flashinfer_cutlass_fused_moe,
    )
    assert install_deterministic_moe_finalize("1")

    # T tokens, D hidden, F expert hidden, E experts, K experts per token.
    # Several experts per token, so the order of the sum matters, and enough
    # tokens per expert that each expert's GEMM spans many tiles.
    T, D, F, E, K = 4096, 1024, 1024, 64, 8
    generator = torch.Generator().manual_seed(2307)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator).to("cuda", torch.bfloat16)

    x_TD = randn(T, D)
    w1_EFD = randn(E, 2 * F, D) / D**0.5
    w2_EDF = randn(E, D, F) / F**0.5
    expert_ids_TK = torch.rand(T, E, generator=generator).topk(K).indices.int().cuda()
    weights_TK = torch.rand(T, K, generator=generator).cuda()

    def run() -> torch.Tensor:
        out_TD = torch.empty_like(x_TD)
        flashinfer_cutlass_moe.flashinfer_cutlass_fused_moe(
            input=x_TD,
            token_selected_experts=expert_ids_TK,
            token_final_scales=weights_TK,
            fc1_expert_weights=w1_EFD,
            fc2_expert_weights=w2_EDF,
            output=out_TD,
            output_dtype=torch.bfloat16,
            quant_scales=[],
        )
        return out_TD

    want_TD = run()
    assert bool(want_TD.isfinite().all()) and float(want_TD.abs().max()) > 0
    # XXX: only SM100 can fail this. There the fused finalize, without the
    # plugin's option, adds expert outputs in a varying order. On SM120 it
    # repeated bit-identically in 20 calls, so the test passes there either way.
    for _ in range(10):
        assert_equal(run(), want_TD)
