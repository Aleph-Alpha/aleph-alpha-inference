# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""HF configs of the Kolibri checkpoints."""

import os
from typing import Any

from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig


class Kolibri1Config(Qwen3MoeConfig):
    """HF config for Kolibri 1 checkpoints."""

    model_type = "kolibri1"

    # transformers generates `__init__` per config subclass; only `__post_init__` is inherited.
    def __post_init__(self, **kwargs: Any) -> None:
        super().__post_init__(**kwargs)
        # DeepGEMM would round our fp32 FP8 scales to powers of two
        os.environ["VLLM_USE_DEEP_GEMM"] = "0"
        os.environ["VLLM_USE_DEEP_GEMM_E8M0"] = "0"


# The class name follows the model name Kolibri 1.1.
class Kolibri1_1Config(Kolibri1Config):  # noqa: N801
    """HF config for Kolibri 1.1 checkpoints, the sparse variant of Kolibri 1."""

    model_type = "kolibri1_1"
