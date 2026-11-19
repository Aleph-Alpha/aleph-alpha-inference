# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""A prefix hit keeps the SWA blocks its bounded replay writes and reads."""

import pytest
import torch

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.single_type_kv_cache_manager import (
    get_manager_for_kv_cache_spec,
    SlidingWindowManager,
)
from vllm.v1.kv_cache_interface import SlidingWindowSpec
from aleph_alpha_inference.model.vllm_replay import (
    BoundedReplaySWAManager,
    BoundedReplaySWASpec,
    register_bounded_replay_spec,
)

BLOCK, WINDOW, REPLAY = 128, 513, 20480


def manager(spec_cls, **kwargs):
    spec = spec_cls(
        block_size=BLOCK,
        num_kv_heads=4,
        head_size=128,
        dtype=torch.bfloat16,
        sliding_window=WINDOW,
        **kwargs,
    )
    pool = BlockPool(4096, enable_caching=True, hash_block_size=BLOCK)
    return get_manager_for_kv_cache_spec(
        spec,
        max_in_flight_tokens=32768,
        max_model_len=65536,
        block_pool=pool,
        enable_caching=True,
        kv_cache_group_id=0,
        scheduler_block_size=BLOCK,
    )


def test_spec_opts_out_of_prefix_caching():
    register_bounded_replay_spec()
    replay = manager(BoundedReplaySWASpec, replay_tokens=REPLAY)
    assert type(replay) is BoundedReplaySWAManager
    assert not replay.kv_cache_spec.prefix_cacheable
    assert replay.kv_cache_spec.prefix_replay_tokens == REPLAY
    assert type(manager(SlidingWindowSpec)) is SlidingWindowManager


@pytest.mark.parametrize("hit", [REPLAY + BLOCK, 3 * REPLAY])
def test_hit_retires_only_blocks_below_the_replay_window(hit):
    register_bounded_replay_spec()
    replay = manager(BoundedReplaySWASpec, replay_tokens=REPLAY)
    # The replay recomputes [hit - REPLAY, hit) and reads W - 1 keys below it.
    num_tokens = hit + 64
    count = replay.get_num_blocks_to_allocate("r", num_tokens, [], hit, hit, num_tokens)
    replay.add_local_computed_blocks("r", [], hit, 0)
    retired = max(hit - REPLAY - WINDOW + 1, 0) // BLOCK
    assert replay.req_to_blocks["r"] == [replay._null_block] * retired
    assert count == (num_tokens + BLOCK - 1) // BLOCK - retired
    # A request without a hit holds every block, as with plain SWA.
    assert replay.get_num_blocks_to_allocate("s", 1024, [], 0, 0, 1024) == 8
