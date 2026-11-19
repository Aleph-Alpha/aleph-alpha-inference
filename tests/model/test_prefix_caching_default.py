# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""Kolibri serving keeps prefix caching off unless the deployment opts in."""

from types import SimpleNamespace

import pytest
from aleph_alpha_inference.model.vllm_config import Kolibri1SparseConfig

from fake_vllm_config import fake_vllm_config


def _config(**serving_options):
    return fake_vllm_config(
        additional_config=serving_options,
        cache_config=SimpleNamespace(
            cache_dtype="bfloat16",
            user_specified_block_size=True,
            block_size=128,
            swa_bounded_replay=False,
            enable_prefix_caching=True,
        ),
        attention_config=SimpleNamespace(backend=None, flash_attn_version=None),
    )


@pytest.mark.parametrize("opt_in", [False, True])
def test_prefix_caching_requires_opt_in(opt_in):
    # Given a deployment that does or does not opt in
    config = _config(prefix_caching=True) if opt_in else _config()
    # When vLLM applies the Kolibri serving defaults
    Kolibri1SparseConfig.verify_and_update_config(config)
    # Then prefix caching follows the serving option
    assert config.cache_config.enable_prefix_caching is opt_in
