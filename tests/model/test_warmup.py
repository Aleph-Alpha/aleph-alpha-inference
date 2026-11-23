# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""Warmup batches stay within the engine's limits."""

from aleph_alpha_inference.model.vllm_warmup import warmup_batches


def test_warmup_batches_fit_a_short_model_length():
    batches = warmup_batches(max_model_len=1024, max_num_reqs=64, max_num_tokens=8192)
    assert batches
    for requests, query, seq_len in batches:
        assert query <= seq_len <= 1024
        assert requests <= 64 and requests * query <= 8192
