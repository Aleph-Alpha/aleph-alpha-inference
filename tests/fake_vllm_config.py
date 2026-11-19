# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""A stand-in for vLLM's config, for tests that do not start an engine."""

from types import SimpleNamespace
from typing import Any

from sparse_hf_config import sparse_hf_config


def fake_vllm_config(
    *,
    hf_text_config: SimpleNamespace | None = None,
    additional_config: dict[str, Any] | None = None,
    max_model_len: int | None = None,
    max_num_seqs: int | None = None,
    max_num_batched_tokens: int | None = None,
    max_cudagraph_capture_size: int | None = None,
    use_ubatching: bool = False,
    **sections: SimpleNamespace,
) -> SimpleNamespace:
    """The parts of a `VllmConfig` that the sparse serving code reads.

    Options with a vLLM default take that default. A limit that is not given
    stays unset, so a test fails when the code reads a limit it did not state.

    Args:
        hf_text_config: The checkpoint's HF config. Defaults to
            `sparse_hf_config()`.
        additional_config: vLLM's `additional_config`, which states the sparse
            serving options. Defaults to no options, as in vLLM.
        max_model_len: Context limit in `model_config`.
        max_num_seqs: Request limit in `scheduler_config`.
        max_num_batched_tokens: Token limit of one step in `scheduler_config`.
        max_cudagraph_capture_size: Largest captured batch in
            `compilation_config`. vLLM's default is `None`.
        use_ubatching: Whether `parallel_config` splits a step into
            overlapping micro-batches.
        **sections: Further config sections by their `VllmConfig` name, e.g.
            `cache_config`.

    Returns:
        The stand-in config.
    """
    model_config = SimpleNamespace(
        hf_text_config=sparse_hf_config() if hf_text_config is None else hf_text_config
    )
    if max_model_len is not None:
        model_config.max_model_len = max_model_len
    scheduler_config = SimpleNamespace()
    if max_num_seqs is not None:
        scheduler_config.max_num_seqs = max_num_seqs
    if max_num_batched_tokens is not None:
        scheduler_config.max_num_batched_tokens = max_num_batched_tokens
    return SimpleNamespace(
        additional_config={} if additional_config is None else additional_config,
        model_config=model_config,
        scheduler_config=scheduler_config,
        compilation_config=SimpleNamespace(
            max_cudagraph_capture_size=max_cudagraph_capture_size
        ),
        parallel_config=SimpleNamespace(use_ubatching=use_ubatching),
        **sections,
    )
