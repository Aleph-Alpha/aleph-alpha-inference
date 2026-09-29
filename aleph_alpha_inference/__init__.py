__version__ = "0.0.0"


def register() -> None:
    """Register the Aleph Alpha models and parsers with vLLM.

    vLLM calls this through the ``vllm.general_plugins`` entry point in every
    process it starts, so it must be idempotent.
    """
    try:
        from transformers import AutoConfig
        from vllm.logger import init_logger
        from vllm.model_executor.models.registry import ModelRegistry
        from vllm.reasoning import ReasoningParserManager
        from vllm.tool_parsers import ToolParserManager

        from aleph_alpha_inference.kolibri1 import Kolibri1Config

        logger = init_logger(__name__)

        # Makes `model_type: "kolibri1"` resolve to Kolibri1Config, for vLLM's
        # get_config (it ends up in AutoConfig.from_pretrained) and for plain
        # transformers consumers alike.
        AutoConfig.register(Kolibri1Config.model_type, Kolibri1Config, exist_ok=True)

        ModelRegistry.register_model(
            "Kolibri1ForCausalLM", "aleph_alpha_inference.kolibri1:Kolibri1ForCausalLM"
        )
        logger.info("Registered Kolibri1ForCausalLM with vLLM")

        ReasoningParserManager.register_lazy_module(
            name="kolibri1",
            module_path="aleph_alpha_inference.reasoning",
            class_name="Kolibri1ParserReasoningAdapter",
        )
        ToolParserManager.register_lazy_module(
            name="kolibri1",
            module_path="aleph_alpha_inference.tool_parser",
            class_name="Kolibri1ToolParser",
        )
        logger.info("Registered kolibri1 reasoning and tool call parsers with vLLM")

    except Exception as e:
        import logging

        logging.getLogger(__name__).error("Failed to register Kolibri 1: %s", e)
        raise


__all__ = [
    "__version__",
    "register",
]
