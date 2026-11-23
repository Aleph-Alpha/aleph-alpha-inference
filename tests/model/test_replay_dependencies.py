# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""Real local-attention dependencies, including KV needed for continuation."""


import pytest
import torch

from aleph_alpha_inference.model.vllm_replay import replay_layout, required_replay_tokens

from sparse_hf_config import FULL_LAYERS, NUM_LAYERS, sparse_hf_config


def run_tail(x, positions, source, program, window):
    inputs = []
    for local in program:
        if local:
            inputs.append(x.clone())
            keys = x
            key_positions = positions
        else:
            keys = source
            key_positions = torch.arange(len(source))
        logits = x @ keys.T / x.shape[-1] ** 0.5
        valid = key_positions[None, :] <= positions[:, None]
        if local:
            valid = valid & (key_positions[None, :] >= positions[:, None] - window + 1)
        probs = logits.masked_fill(~valid, float("-inf")).softmax(-1)
        x = torch.tanh(x + 0.37 * (probs @ keys))
    return x, inputs


@pytest.mark.parametrize("window", [3, 5])
@pytest.mark.parametrize(
    "program", [(True, False, True, True, False), (False, True, False, True)]
)
def test_pruned_tail_preserves_last_output_and_next_decode_cache(window, program):
    torch.manual_seed(81)
    x = torch.randn(37, 4, dtype=torch.float64)
    source = torch.randn_like(x)
    positions = torch.arange(len(x))
    full, caches = run_tail(x, positions, source, program, window)
    keep = required_replay_tokens(window, sum(program))
    tail, tail_caches = run_tail(x[-keep:], positions[-keep:], source, program, window)
    # Truncation changes matrix/reduction shapes, allowing FP64 roundoff.
    torch.testing.assert_close(tail[-1], full[-1], rtol=0, atol=1e-12)
    for a, b in zip(caches, tail_caches):
        torch.testing.assert_close(
            a[-window + 1 :], b[-window + 1 :], rtol=0, atol=1e-12
        )
    # A single window is insufficient when multiple local layers are stacked.
    short, _ = run_tail(x[-window:], positions[-window:], source, program, window)
    assert (short[-1] - full[-1]).abs().max() > 1e-7


def test_kolibri_bounds_and_invalid_inputs():
    assert required_replay_tokens(513, 20) == 10241
    assert required_replay_tokens(513, 8) == 4097
    assert required_replay_tokens(513, 0) == 1
    assert required_replay_tokens(1, 20) == 1
    with pytest.raises(ValueError):
        required_replay_tokens(0, 20)
    with pytest.raises(ValueError):
        required_replay_tokens(513, -1)


@pytest.mark.parametrize("window", [3, 5])
@pytest.mark.parametrize(
    "program", [(True, False, True, True, False), (False, True, False, True)]
)
def test_prefix_replay_rebuilds_cache_for_tokens_after_the_hit(window, program):
    torch.manual_seed(83)
    x = torch.randn(41, 4, dtype=torch.float64)
    source = torch.randn_like(x)
    positions = torch.arange(len(x))
    hit = 30
    full, caches = run_tail(x, positions, source, program, window)
    # The hit's KV of shared-global layers stays cached; the replay rebuilds
    # local KV from `start`, attending to no local key below it.
    start = hit - (required_replay_tokens(window, sum(program)) - 1)
    out, replay_caches = run_tail(x[start:], positions[start:], source, program, window)
    # Truncation changes matrix/reduction shapes, allowing FP64 roundoff.
    torch.testing.assert_close(out[hit - start :], full[hit:], rtol=0, atol=1e-12)
    for a, b in zip(caches, replay_caches):
        torch.testing.assert_close(
            a[hit - window + 1 :], b[hit - window + 1 - start :], rtol=0, atol=1e-12
        )
    short, _ = run_tail(x[start + 1 :], positions[start + 1 :], source, program, window)
    if sum(program) > 1:
        assert (short[hit - start - 1 :] - full[hit:]).abs().max() > 1e-7


def sparse_config(producers):
    """A config whose sparse layers read the K/V of the last of the one-based
    `producers` up to them."""
    kv_source_layers = list(range(NUM_LAYERS))
    for layer in FULL_LAYERS:
        kv_source_layers[layer] = max(p - 1 for p in producers if p - 1 <= layer)
    return sparse_hf_config(sliding_window=513, kv_source_layers=kv_source_layers)


@pytest.mark.parametrize(
    "producers,layout",
    [((5, 20, 25, 30), (29, 8193, 20480)), ((5, 15, 20, 25), (24, 10241, 20480))],
)
def test_kolibri_replay_layouts(producers, layout):
    actual = replay_layout(sparse_config(producers))
    assert (
        actual.decoder_start,
        actual.decoder_window,
        actual.prefix_replay_tokens,
    ) == layout
