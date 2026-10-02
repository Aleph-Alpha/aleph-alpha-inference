# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH

import os
from typing import TYPE_CHECKING

from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from vllm.model_executor.models.config import VerifyAndUpdateConfig

if TYPE_CHECKING:
    from vllm.config import ModelConfig


class Kolibri1Config(Qwen3MoeConfig):
    model_type = "kolibri1"


class Kolibri1ForCausalLMConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_model_config(model_config: "ModelConfig") -> None:
        # Kolibri 1 FP8 weights carry fp32 block scales, which DeepGEMM would
        # round to powers of two (UE8M0).
        os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
        os.environ.setdefault("VLLM_USE_DEEP_GEMM_E8M0", "0")
