# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""Serving options come from vLLM's `additional_config`, with defaults."""

from typing import Any

import pytest
from aleph_alpha_inference.kernels.serving_options import (
    sparse_serving_options,
    SparseServingOptions,
)

from fake_vllm_config import fake_vllm_config


def test_defaults_apply_without_stated_options() -> None:
    options = sparse_serving_options(fake_vllm_config(additional_config={}))
    assert options == SparseServingOptions(prefix_caching=False)


def test_additional_config_states_the_options() -> None:
    # Entries of other consumers of `additional_config` are left alone.
    additional_config = {"prefix_caching": True, "gdn_prefill_backend": "auto"}
    options = sparse_serving_options(
        fake_vllm_config(additional_config=additional_config)
    )
    assert options == SparseServingOptions(prefix_caching=True)


@pytest.mark.parametrize("name, value", [("prefix_caching", "false")])
def test_unsupported_values_are_rejected(name: str, value: Any) -> None:
    with pytest.raises(ValueError, match=name):
        sparse_serving_options(fake_vllm_config(additional_config={name: value}))
