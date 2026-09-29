"""Unit tests for the Kolibri 1 reasoning parser.

These drive the registered ``ReasoningParser`` adapters through the same
``DelegatingParser`` streaming path vLLM's chat serving layer uses, with a
tiny mock tokenizer that only knows the delimiter tokens; every other word
is one plain text token.

Prompts are written the way the Kolibri chat template renders them. With
thinking disabled the template prefills the closed, empty
``<think>\n\n</think>\n\n`` block right after ``<|im_start|>assistant\n``,
so the model starts at the answer. The tests at the end render the real
template, checked in as ``kolibri1_chat_template.jinja``, instead of
hand-written prompts.
"""

import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _make_mock_tokenizer() -> MagicMock:
    vocab = {
        "<think>": 50,
        "</think>": 51,
        "<tool_call>": 60,
        "</tool_call>": 61,
        "<|im_start|>": 70,
        "<|im_end|>": 71,
    }
    id_to_text = {v: k for k, v in vocab.items()}
    tokenizer = MagicMock()
    tokenizer.get_vocab.return_value = dict(vocab)
    tokenizer.decode.side_effect = lambda ids: "".join(
        id_to_text.get(i, f"<{i}>") for i in ids
    )
    tokenizer.all_special_tokens = list(vocab)
    tokenizer.all_special_ids = list(vocab.values())
    return tokenizer


def _token_ids(tokenizer, *tokens: str) -> list[int]:
    """One token id per word; words outside the vocab share a text id."""
    vocab = tokenizer.get_vocab()
    return [vocab.get(token, 100) for token in tokens]


def _make_request() -> MagicMock:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )

    request = MagicMock(spec=ChatCompletionRequest)
    request.tools = []
    request.tool_choice = "auto"
    request.include_reasoning = True
    return request


@pytest.fixture
def kolibri1_adapter_cls(register_plugin):
    from vllm.reasoning import ReasoningParserManager

    return ReasoningParserManager.get_reasoning_parser("kolibri1")


@pytest.fixture
def tokenizer():
    return _make_mock_tokenizer()


def _make_parser(reasoning_adapter_cls, tokenizer, **chat_template_kwargs):
    """A DelegatingParser built like vLLM's serving layer does.

    The serving layer hands the request's ``chat_template_kwargs`` to the
    reasoning parser; that is the only way it learns whether thinking is on.
    """
    from vllm.parser.abstract_parser import DelegatingParser

    from aleph_alpha_inference.tool_parser import Kolibri1ToolParser

    class _Kolibri1DelegatingParser(DelegatingParser):
        reasoning_parser_cls = reasoning_adapter_cls
        tool_parser_cls = Kolibri1ToolParser

    return _Kolibri1DelegatingParser(
        tokenizer, chat_template_kwargs=chat_template_kwargs
    )


@pytest.fixture
def parser(kolibri1_adapter_cls, tokenizer):
    return _make_parser(kolibri1_adapter_cls, tokenizer)


def _stream(parser, tokenizer, prompt: list[str], generation: list[str]):
    """Feed the generation into the parser one token at a time.

    Returns the (reasoning, content) pair of each delta; ``None`` where the
    parser emitted nothing.
    """
    request = _make_request()
    prompt_token_ids = _token_ids(tokenizer, *prompt)
    deltas = []
    for i, token in enumerate(generation):
        delta = parser.parse_delta(
            token,
            _token_ids(tokenizer, token),
            request,
            prompt_token_ids=prompt_token_ids,
            finished=i == len(generation) - 1,
        )
        deltas.append(None if delta is None else (delta.reasoning, delta.content))
    return deltas


def test_single_turn_prompt_is_not_reasoning_end(kolibri1_adapter_cls, tokenizer):
    adapter = kolibri1_adapter_cls(tokenizer)
    prompt = ["<|im_start|>", "user", "<|im_end|>", "<|im_start|>", "assistant"]
    assert not adapter.is_reasoning_end(_token_ids(tokenizer, *prompt))


def test_single_turn_reasoning_streams_as_reasoning(parser, tokenizer):
    prompt = ["<|im_start|>", "user", "<|im_end|>", "<|im_start|>", "assistant"]
    generation = ["<think>", "thinking", "</think>", "answer"]
    assert _stream(parser, tokenizer, prompt, generation) == [
        None,
        ("thinking", None),
        None,
        (None, "answer"),
    ]


def test_prompt_after_tool_call_is_not_reasoning_end(kolibri1_adapter_cls, tokenizer):
    """The chat template keeps the reasoning of assistant turns after the
    last user message, so after a tool call the earlier turn's ``</think>``
    is still in the prompt. Reasoning must still count as not yet ended:
    only the current turn decides.
    """
    adapter = kolibri1_adapter_cls(tokenizer)
    prompt = [
        "<|im_start|>", "user", "<|im_end|>",
        "<|im_start|>", "assistant", "<think>", "reasoning", "</think>",
        "<tool_call>", "call", "</tool_call>", "<|im_end|>",
        "<|im_start|>", "user", "result", "<|im_end|>",
        "<|im_start|>", "assistant",
    ]  # fmt: skip
    assert not adapter.is_reasoning_end(_token_ids(tokenizer, *prompt))


def test_reasoning_after_tool_call_streams_as_reasoning(parser, tokenizer):
    """Same case as above, seen from the streaming side: the new turn's
    thinking must stream as ``reasoning``, not ``content``."""
    prompt = [
        "<|im_start|>", "user", "<|im_end|>",
        "<|im_start|>", "assistant", "<think>", "reasoning", "</think>",
        "<tool_call>", "call", "</tool_call>", "<|im_end|>",
        "<|im_start|>", "user", "result", "<|im_end|>",
        "<|im_start|>", "assistant",
    ]  # fmt: skip
    generation = ["<think>", "thinking", "</think>", "answer"]
    assert _stream(parser, tokenizer, prompt, generation) == [
        None,
        ("thinking", None),
        None,
        (None, "answer"),
    ]


# ---------------------------------------------------------------------------
# Thinking switch: ``reasoning_effort`` and ``enable_thinking``
#
# The template gives ``reasoning_effort`` precedence and falls back to
# ``enable_thinking`` only when it is absent or ``None``, and only a literal
# ``false`` turns thinking off.
# ---------------------------------------------------------------------------

THINKING_OFF_KWARGS = [
    {"enable_thinking": False},
    {"reasoning_effort": None, "enable_thinking": False},
    {"reasoning_effort": "none"},
    {"reasoning_effort": "none", "enable_thinking": True},
]

THINKING_ON_KWARGS = [
    {},
    {"reasoning_effort": None},
    {"enable_thinking": True},
    {"enable_thinking": None},
    {"reasoning_effort": "high"},
    {"reasoning_effort": "low", "enable_thinking": False},
]


def _kwargs_id(kwargs: dict) -> str:
    return ",".join(f"{k}={v}" for k, v in kwargs.items()) or "default"


@pytest.mark.parametrize("kwargs", THINKING_OFF_KWARGS, ids=_kwargs_id)
def test_thinking_off_kwargs(kwargs):
    from aleph_alpha_inference.reasoning import thinking_enabled

    assert not thinking_enabled(kwargs)


@pytest.mark.parametrize("kwargs", THINKING_ON_KWARGS, ids=_kwargs_id)
def test_thinking_on_kwargs(kwargs):
    from aleph_alpha_inference.reasoning import thinking_enabled

    assert thinking_enabled(kwargs)


@pytest.mark.parametrize("kwargs", THINKING_OFF_KWARGS, ids=_kwargs_id)
def test_thinking_off_non_streaming_is_content(kolibri1_adapter_cls, tokenizer, kwargs):
    """Non-streaming never sees the prompt, so only the request kwargs tell
    the parser that the block is already closed. Started in the reasoning
    state it would file the whole answer as reasoning."""
    adapter = kolibri1_adapter_cls(tokenizer, chat_template_kwargs=kwargs)
    assert adapter.extract_reasoning("The answer.", _make_request()) == (
        None,
        "The answer.",
    )


@pytest.mark.parametrize("kwargs", THINKING_ON_KWARGS, ids=_kwargs_id)
def test_thinking_on_non_streaming_splits_reasoning(
    kolibri1_adapter_cls, tokenizer, kwargs
):
    adapter = kolibri1_adapter_cls(tokenizer, chat_template_kwargs=kwargs)
    reasoning, content = adapter.extract_reasoning(
        "<think>\nlet me see\n</think>\n\nThe answer.", _make_request()
    )
    assert (reasoning.strip(), content.strip()) == ("let me see", "The answer.")


@pytest.mark.parametrize("kwargs", THINKING_ON_KWARGS, ids=_kwargs_id)
def test_thinking_on_non_streaming_truncated_is_reasoning(
    kolibri1_adapter_cls, tokenizer, kwargs
):
    """Cut off mid-think, the output is reasoning only; this pins the parser
    to the reasoning state for every thinking-on combination."""
    adapter = kolibri1_adapter_cls(tokenizer, chat_template_kwargs=kwargs)
    assert adapter.extract_reasoning("<think>\nstill", _make_request()) == (
        "\nstill",
        None,
    )


# ---------------------------------------------------------------------------
# The real chat template. Rendered prompts are split into the delimiter
# tokens, which are single tokens in the real tokenizer too, and the text in
# between, one plain token per stretch, so the mock tokenizer can id them.
# ---------------------------------------------------------------------------

CHAT_TEMPLATE = (Path(__file__).parent / "kolibri1_chat_template.jinja").read_text()
ALL_KWARGS = THINKING_OFF_KWARGS + THINKING_ON_KWARGS
PREFILLED_BLOCK = "<think>\n\n</think>\n\n"
USER_MESSAGES = [{"role": "user", "content": "hi"}]
TOOL_LOOP_MESSAGES = [
    *USER_MESSAGES,
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": "lookup", "arguments": {}}}],
    },
    {"role": "tool", "content": "result"},
]


def _render(kwargs, messages=USER_MESSAGES) -> str:
    from transformers.utils.chat_template_utils import render_jinja_template

    rendered, _ = render_jinja_template(
        conversations=[messages],
        chat_template=CHAT_TEMPLATE,
        add_generation_prompt=True,
        **kwargs,
    )
    return rendered[0]


def _tokens(tokenizer, text: str) -> list[str]:
    """Split rendered text into delimiter tokens and the text between them."""
    delimiters = "|".join(re.escape(token) for token in tokenizer.get_vocab())
    return [piece for piece in re.split(f"({delimiters})", text) if piece]


@pytest.mark.parametrize("kwargs", ALL_KWARGS, ids=_kwargs_id)
def test_template_prefills_block_iff_thinking_off(kwargs):
    from aleph_alpha_inference.reasoning import thinking_enabled

    prompt = _render(kwargs)
    assert prompt.endswith("<|im_start|>assistant\n" + PREFILLED_BLOCK) == (
        not thinking_enabled(kwargs)
    )


@pytest.mark.parametrize("kwargs", ALL_KWARGS, ids=_kwargs_id)
def test_template_gate_agrees_with_parser(kolibri1_adapter_cls, tokenizer, kwargs):
    """The structured-output gate: with thinking off, a grammar applies from
    the first token because the prefilled block already closed the reasoning."""
    from aleph_alpha_inference.reasoning import thinking_enabled

    adapter = kolibri1_adapter_cls(tokenizer, chat_template_kwargs=kwargs)
    prompt_ids = _token_ids(tokenizer, *_tokens(tokenizer, _render(kwargs)))
    assert adapter.is_reasoning_end(prompt_ids) == (not thinking_enabled(kwargs))


@pytest.mark.parametrize(
    "messages", [USER_MESSAGES, TOOL_LOOP_MESSAGES], ids=["single", "tool_loop"]
)
@pytest.mark.parametrize("kwargs", THINKING_OFF_KWARGS, ids=_kwargs_id)
def test_template_thinking_off_streams_as_content(
    kolibri1_adapter_cls, tokenizer, kwargs, messages
):
    parser = _make_parser(kolibri1_adapter_cls, tokenizer, **kwargs)
    prompt = _tokens(tokenizer, _render(kwargs, messages))
    assert _stream(parser, tokenizer, prompt, ["The", " answer."]) == [
        (None, "The"),
        (None, " answer."),
    ]


@pytest.mark.parametrize(
    "messages", [USER_MESSAGES, TOOL_LOOP_MESSAGES], ids=["single", "tool_loop"]
)
@pytest.mark.parametrize("kwargs", THINKING_ON_KWARGS, ids=_kwargs_id)
def test_template_thinking_on_streams_reasoning_then_content(
    kolibri1_adapter_cls, tokenizer, kwargs, messages
):
    parser = _make_parser(kolibri1_adapter_cls, tokenizer, **kwargs)
    prompt = _tokens(tokenizer, _render(kwargs, messages))
    generation = ["<think>", "thinking", "</think>", "answer"]
    assert _stream(parser, tokenizer, prompt, generation) == [
        None,
        ("thinking", None),
        None,
        (None, "answer"),
    ]
