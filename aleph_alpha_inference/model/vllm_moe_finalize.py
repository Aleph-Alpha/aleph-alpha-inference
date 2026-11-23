# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""Deterministic finalize for vLLM's FlashInfer CUTLASS MoE.

On B200, vLLM serves unquantized BF16 MoE with FlashInfer's
`cutlass_fused_moe`. By default it adds up each token's top-k expert outputs
in a fused epilogue with atomics, so the same request gives different logprobs
from run to run (vLLM issue 54945). Passing `use_fused_finalize=False` makes
FlashInfer add them in a fixed order. vLLM 0.33 does not pass the option, so
this module wraps the function vLLM calls.

On Kolibri 1 sparse on one B200, the unfused finalize cost 0 to 4% of offline
throughput; Triton MoE, the other deterministic choice, cost 6 to 11%.

FlashInfer tunes the unfused finalize's GEMMs separately. With one process,
vLLM tunes them at startup. With several (TP, PP, EP or multi-node), vLLM reads
FlashInfer's autotune cache file instead; a file tuned for the fused finalize
then counts as complete, and the unfused finalize runs untuned, which cost 18
to 21% of MoE time. Start such deployments with a fresh
`VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR`.

XXX: remove once vLLM lets us choose the finalize (vLLM PR 54948 adds
`VLLM_FLASHINFER_MOE_FUSED_FINALIZE`) and the serving image sets it. Installed
only on the vLLM minor version it was checked against.
"""

import functools
from typing import Any

import vllm
from vllm.logger import init_logger

from aleph_alpha_inference.env import env_choice

logger = init_logger(f"vllm.{__name__}")

# "1" (default) passes `use_fused_finalize=False`, "0" keeps vLLM's call.
MODE_ENV = "KOLIBRI_VLLM_DETERMINISTIC_MOE"
MODES = ("0", "1")
SUPPORTED_VLLM_VERSION = "0.33."


def install_deterministic_moe_finalize(mode: str | None = None) -> bool:
    """Make vLLM's FlashInfer CUTLASS MoE add expert outputs in a fixed order.

    Args:
        mode: One of `MODES`; read from `KOLIBRI_VLLM_DETERMINISTIC_MOE` when
            None, defaulting to "1", also for a value that is not a mode.

    Returns:
        Whether vLLM's FlashInfer MoE call passes `use_fused_finalize=False`
        after the call. Installing twice is a no-op. Skipped with a log line on
        other vLLM versions. Patches
        `vllm.model_executor.layers.fused_moe.experts.flashinfer_cutlass_moe`
        process-wide, for every model.
    """
    if mode is None:
        mode = env_choice(MODE_ENV, MODES, "1")
    if mode not in MODES:
        raise ValueError(f"{MODE_ENV} must be one of {MODES}, got {mode!r}")
    if mode == "0":
        logger.info("FlashInfer MoE keeps its fused finalize: %s=0", MODE_ENV)
        return False
    if not vllm.__version__.startswith(SUPPORTED_VLLM_VERSION):
        logger.warning(
            "Deterministic FlashInfer MoE finalize skipped: written for vLLM "
            "%sx, found %s",
            SUPPORTED_VLLM_VERSION,
            vllm.__version__,
        )
        return False
    # Imported only on the checked vLLM version: another version may have
    # moved the module.
    from vllm.model_executor.layers.fused_moe.experts import (
        flashinfer_cutlass_moe as module,
    )

    original = module.flashinfer_cutlass_fused_moe
    if getattr(original, "_kolibri_unfused_finalize", False):
        return True

    @functools.wraps(original)
    def flashinfer_cutlass_fused_moe(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("use_fused_finalize", False)
        return original(*args, **kwargs)

    flashinfer_cutlass_fused_moe._kolibri_unfused_finalize = True
    module.flashinfer_cutlass_fused_moe = flashinfer_cutlass_fused_moe
    logger.info("FlashInfer MoE uses its deterministic, unfused finalize")
    return True
