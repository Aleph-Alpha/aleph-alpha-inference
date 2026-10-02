# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""Reasoning parsers for the Kolibri model family.

The chat template and the reasoning parser split one job. The template
decides whether the generation prompt already contains the closed, empty
``<think>\\n\\n</think>\\n\\n`` block (thinking off) or stops right after
``<|im_start|>assistant\\n`` so the model opens the block itself (thinking
on). The parser decides, token by token, whether generated text is reasoning
or answer, and its starting state must match what the template rendered.

The Kolibri template switches thinking off when
``reasoning_effort`` is ``"none"``; only when no ``reasoning_effort`` is
given does it fall back to ``enable_thinking``. vLLM's stock Qwen3 parser
reads ``enable_thinking`` alone. vLLM derives that from a top-level
``reasoning_effort`` only when the request sets no ``enable_thinking``, so
the stock parser goes wrong when the effort arrives in
``chat_template_kwargs`` or contradicts an explicit ``enable_thinking``. It
then starts in the wrong state and, on the non-streaming path where it never
sees the prompt, files the answer as reasoning or the reasoning as answer.
``Kolibri1Parser`` derives the starting state the same way the template
does.

This covers the generation prompt only: a continued final assistant message
(``continue_final_message``) always gets the closed block, whatever the
switch says, and the parser does not account for that.
"""

from collections.abc import Mapping
from typing import Any

from vllm.parser.engine.adapters import ParserEngineReasoningAdapter
from vllm.parser.qwen3 import Qwen3Parser


def thinking_enabled(chat_template_kwargs: Mapping[str, Any] | None) -> bool:
    """Whether the Kolibri chat template renders the prompt for thinking.

    Mirrors the template's switch: a ``reasoning_effort`` other than ``None``
    wins, and only ``"none"`` disables thinking. Without it, only a literal
    ``enable_thinking: false`` disables thinking, as the template tests
    ``enable_thinking is false``.
    """
    kwargs = chat_template_kwargs or {}
    effort = kwargs.get("reasoning_effort")
    if effort is not None:
        return effort != "none"
    return kwargs.get("enable_thinking") is not False


class Kolibri1Parser(Qwen3Parser):
    """Qwen3 grammar with the starting state chosen like the Kolibri 1 template."""

    def __init__(self, tokenizer, tools=None, **kwargs) -> None:
        chat_kwargs = dict(kwargs.get("chat_template_kwargs") or {})
        chat_kwargs["enable_thinking"] = thinking_enabled(chat_kwargs)
        kwargs["chat_template_kwargs"] = chat_kwargs
        super().__init__(tokenizer, tools, **kwargs)


class Kolibri1ParserReasoningAdapter(ParserEngineReasoningAdapter):
    _parser_engine_cls = Kolibri1Parser
