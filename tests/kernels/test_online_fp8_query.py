# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.config import set_current_vllm_config, VllmConfig
from vllm.model_executor.layers.quantization.online.fp8 import (
    Fp8PerBlockOnlineLinearMethod,
)
from aleph_alpha_inference.model.vllm_model import NativeSparseAttention, QueryWeightView

from tensor_asserts import assert_equal


@pytest.mark.gpu
@pytest.mark.parametrize("tokens", [1, 128, 1024])
def test_online_fp8_query_matches_full_projection(tokens):
    torch.manual_seed(42)
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with (
            patch(
                "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
                return_value=1,
            ),
            patch(
                "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
                return_value=0,
            ),
            set_current_vllm_config(VllmConfig()),
            torch.device("cuda"),
            patch(
                "vllm.model_executor.layers.quantization.online.fp8.get_current_vllm_config",
                return_value=SimpleNamespace(
                    model_config=SimpleNamespace(dtype=torch.bfloat16)
                ),
            ),
        ):
            method = Fp8PerBlockOnlineLinearMethod()
            projection = torch.nn.Module()
            method.create_weights(
                projection, 2560, [6144, 512, 512], 2560, 7168, torch.bfloat16
            )
            projection.weight = torch.nn.Parameter(
                torch.randn(7168, 2560, dtype=torch.bfloat16) * 0.02,
                requires_grad=False,
            )
            method.process_weights_after_loading(projection)
            projection.quant_method = method
            attention = SimpleNamespace(
                qkv_proj=projection,
                q_size=6144,
                _query_weight_view=QueryWeightView(projection, 6144),
            )
            inputs = torch.randn(tokens, 2560, dtype=torch.bfloat16)
            actual = NativeSparseAttention._query_projection(attention, inputs)
            expected = method.apply(projection, inputs)[:, :6144]
            assert_equal(actual, expected)
    finally:
        torch.set_default_dtype(original_dtype)
