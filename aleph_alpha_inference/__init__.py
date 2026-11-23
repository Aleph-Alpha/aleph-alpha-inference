# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""vLLM plugin that serves the Kolibri models."""

from importlib.metadata import version

__version__ = version("aleph-alpha-inference")
del version

# Where vLLM imports the models and parsers from. The paths follow this
# package, wherever it is installed.
DENSE_MODEL = f"{__name__}.kolibri1:Kolibri1ForCausalLM"
SPARSE_MODEL = f"{__name__}.model.vllm_model:Kolibri1_1ForCausalLM"
REASONING_MODULE = f"{__name__}.reasoning"
TOOL_PARSER_MODULE = f"{__name__}.tool_parser"


def _patch_vllm_process() -> None:
    """Install the plugin's changes to vLLM itself, which apply to every model
    in the process. See "Changes to vLLM" in the README for their switches."""
    from aleph_alpha_inference.model.vllm_decode_fast_path import (
        install_decode_fast_path,
    )
    from aleph_alpha_inference.model.vllm_gc import configure_gc
    from aleph_alpha_inference.model.vllm_moe_finalize import (
        install_deterministic_moe_finalize,
    )

    # vLLM loads general plugins in the engine-core process too, where the
    # scheduler's KV-cache manager runs.
    install_decode_fast_path()
    configure_gc()
    install_deterministic_moe_finalize()


def register() -> None:
    """Register the Kolibri models, their parsers and the plugin's changes to
    vLLM.

    vLLM calls this through the `vllm.general_plugins` entry point, in every
    process. Calling it again changes nothing.
    """
    try:
        from transformers import AutoConfig
        from vllm.logger import init_logger
        from vllm.model_executor.models.config import MODELS_CONFIG_MAP
        from vllm.model_executor.models.registry import ModelRegistry
        from vllm.reasoning import ReasoningParserManager
        from vllm.tool_parsers import ToolParserManager

        from aleph_alpha_inference import repetition
        from aleph_alpha_inference.config import Kolibri1_1Config, Kolibri1Config
        from aleph_alpha_inference.cuda_toolkit import prepare_cuda_toolkit
        from aleph_alpha_inference.model.vllm_config import Kolibri1SparseConfig
        from aleph_alpha_inference.model.vllm_replay import register_bounded_replay_spec

        logger = init_logger(__name__)

        # Before anything compiles: the plugin's kernels and FlashInfer's
        # build on first use, and need a toolkit they can find and link with.
        prepare_cuda_toolkit()

        # Makes `model_type: "kolibri1"` and "kolibri1_1" resolve to their
        # configs, for vLLM's get_config (it ends up in
        # AutoConfig.from_pretrained) and for plain transformers consumers
        # alike. `exist_ok` keeps it idempotent, since process0, the engine
        # core and every worker each load plugins.
        AutoConfig.register(Kolibri1Config.model_type, Kolibri1Config, exist_ok=True)
        AutoConfig.register(
            Kolibri1_1Config.model_type, Kolibri1_1Config, exist_ok=True
        )

        ModelRegistry.register_model("Kolibri1ForCausalLM", DENSE_MODEL)
        logger.info("Registered Kolibri1ForCausalLM with vLLM")

        for name in ("kolibri1", "kolibri"):
            ReasoningParserManager.register_lazy_module(
                name=name,
                module_path=REASONING_MODULE,
                class_name="Kolibri1ParserReasoningAdapter",
            )
            ToolParserManager.register_lazy_module(
                name=name,
                module_path=TOOL_PARSER_MODULE,
                class_name="Kolibri1ToolParser",
            )
        logger.info("Registered kolibri reasoning and tool parsers with vLLM")

        # Sparse block attention kernels (SM100, SM90) and a bounded-replay
        # sliding-window KV cache on top of Kolibri 1.
        register_bounded_replay_spec()
        ModelRegistry.register_model("Kolibri1_1ForCausalLM", SPARSE_MODEL)
        # vLLM runs these hooks while building the config, before it picks
        # the attention backend and KV cache layout.
        MODELS_CONFIG_MAP["Kolibri1_1ForCausalLM"] = Kolibri1SparseConfig
        logger.info("Registered Kolibri1_1ForCausalLM with vLLM")

        _patch_vllm_process()

        stride = repetition.stride_from_env()
        repetition.patch_vllm(stride)
        logger.info("Repetition detection checks every %d tokens", stride)
    except Exception as e:
        import logging

        logging.getLogger(__name__).error("Failed to register Kolibri: %s", e)
        raise


__all__ = [
    "__version__",
    "register",
]
