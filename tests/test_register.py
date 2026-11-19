# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
from vllm.model_executor.models.config import MODELS_CONFIG_MAP
from vllm.model_executor.models.registry import ModelRegistry
import aleph_alpha_inference
from aleph_alpha_inference.model.vllm_config import Kolibri1SparseConfig


def test_register_is_idempotent():
    aleph_alpha_inference.register()
    # vLLM loads plugins in every process, so registering again must be a no-op.
    aleph_alpha_inference.register()

    # Checkpoint configs name these architectures, so dropping one breaks
    # loading.
    archs = ModelRegistry.get_supported_archs()
    assert "Kolibri1ForCausalLM" in archs
    assert "Kolibri1_1ForCausalLM" in archs
    assert MODELS_CONFIG_MAP["Kolibri1_1ForCausalLM"] is Kolibri1SparseConfig
