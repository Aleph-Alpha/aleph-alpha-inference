# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""The decode fast path leaves vLLM's KV-cache bookkeeping unchanged.

A small scheduler drives a real `KVCacheManager` with full-attention and
sliding-window groups of different block sizes through chunked prefill,
prefix hits, many decode steps, preemption and request free. A manager with
the fast path must return the same blocks and keep the same block tables and
prefix cache as a stock one, step for step.
"""

import inspect
import math
import random

import pytest
import torch
import vllm
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    SlidingWindowSpec,
)
from vllm.v1.request import Request, RequestStatus
from aleph_alpha_inference.model.vllm_decode_fast_path import (
    install_decode_fast_path,
    SUPPORTED_VLLM_VERSION,
)
from aleph_alpha_inference.model.vllm_replay import (
    BoundedReplaySWASpec,
    register_bounded_replay_spec,
)

HASH_BLOCK = 16
MAX_MODEL_LEN = 4096
CHUNK = 100
MAX_STEPS = 5000


class StockKVCacheManager(KVCacheManager):
    """vLLM's manager without the fast path, even after `register()` installed
    it on `KVCacheManager` earlier in the session."""

    allocate_slots = inspect.unwrap(KVCacheManager.allocate_slots)
    free = inspect.unwrap(KVCacheManager.free)
    pop_blocks_for_free = inspect.unwrap(KVCacheManager.pop_blocks_for_free)


class FastKVCacheManager(StockKVCacheManager):
    pass


class CheckedKVCacheManager(StockKVCacheManager):
    pass


install_decode_fast_path(FastKVCacheManager, mode="1")
install_decode_fast_path(CheckedKVCacheManager, mode="check")


def test_fast_path_installs_on_the_locked_vllm():
    """A vLLM bump fails here until someone re-checks the fast path against it."""
    assert vllm.__version__.startswith(SUPPORTED_VLLM_VERSION), (
        f"vLLM {vllm.__version__}: check that the fast path still matches "
        "KVCacheManager, then update SUPPORTED_VLLM_VERSION"
    )
    assert "_kolibri_fast_path_mode" in vars(FastKVCacheManager)


def attention(spec_cls, block_size, **kwargs):
    return spec_cls(
        block_size=block_size,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.bfloat16,
        **kwargs,
    )


def sliding_window_groups():
    """Full attention and plain SWA groups of different block sizes and windows."""
    return [
        attention(FullAttentionSpec, 64),
        attention(SlidingWindowSpec, 32, sliding_window=70),
        attention(SlidingWindowSpec, 16, sliding_window=17),
        attention(SlidingWindowSpec, 48, sliding_window=100),
    ]


def bounded_replay_groups():
    """Kolibri's layout: full attention plus SWA groups with bounded replay."""
    return [
        attention(FullAttentionSpec, 64),
        attention(BoundedReplaySWASpec, 32, sliding_window=65, replay_tokens=129),
        attention(BoundedReplaySWASpec, 16, sliding_window=65, replay_tokens=129),
    ]


def make_manager(manager_cls, specs, num_blocks, enable_caching):
    register_bounded_replay_spec()
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec([f"layer{i}"], spec) for i, spec in enumerate(specs)
        ],
        hash_block_size=HASH_BLOCK,
    )
    return manager_cls(
        config,
        max_model_len=MAX_MODEL_LEN,
        scheduler_block_size=math.lcm(*(spec.block_size for spec in specs)),
        hash_block_size=HASH_BLOCK,
        enable_caching=enable_caching,
    )


def make_requests(seed, num_requests):
    rng = random.Random(seed)
    shared = [rng.randrange(1000) for _ in range(300)]
    requests = []
    for i in range(num_requests):
        length = rng.randrange(40, 700)
        # Every other request shares a prefix, so later ones hit the cache.
        prompt = shared[:length] if i % 2 else []
        prompt += [rng.randrange(1000) for _ in range(length - len(prompt))]
        requests.append(
            (f"req{i}", prompt, rng.randrange(150, 600), rng.randrange(0, 40))
        )
    return requests


class Scheduler:
    """Schedules requests in admission order, the way vLLM's scheduler calls
    the KV-cache manager.

    With `lag` > 0, a step's tokens stay in flight (`num_in_flight_tokens`)
    for `lag` more steps and decode samples arrive late, as under async
    scheduling. A request whose allocation fails preempts the newest running
    request, as vLLM does.
    """

    def __init__(self, manager, requests, lag):
        init_none_hash(sha256)
        hasher = get_request_block_hasher(HASH_BLOCK, sha256)
        self.manager = manager
        self.replay_tokens = max(
            (
                group.kv_cache_spec.prefix_replay_tokens
                for group in manager.coordinator.single_type_managers
            ),
            default=0,
        )
        self.lag = lag
        self.waiting = []
        self.arrivals = {}
        for request_id, prompt, max_tokens, arrival in requests:
            request = Request(
                request_id,
                prompt,
                SamplingParams(max_tokens=max_tokens),
                None,
                block_hasher=hasher,
            )
            self.arrivals.setdefault(arrival, []).append(request)
        self.running = []
        self.in_flight = []
        self.log = []
        self.step = 0

    def run(self):
        while self.arrivals or self.waiting or self.running or any(self.in_flight):
            self.waiting += self.arrivals.pop(self.step, [])
            self.schedule()
            self.step += 1
            self.finish_steps(self.lag)
            assert self.step < MAX_STEPS, "requests stopped making progress"
        return self.log

    def target(self, request):
        return request.num_tokens + request.num_output_placeholders

    def schedule(self):  # noqa: C901 - mirrors the scheduler loop it stands in for
        self.scheduled = scheduled = []
        for request in list(self.running):
            if request not in self.running:
                continue
            num_new = min(CHUNK, self.target(request) - request.num_computed_tokens)
            if num_new <= 0:
                continue
            while True:
                blocks = self.manager.allocate_slots(request, num_new)
                self.record(request, blocks)
                if blocks is not None:
                    break
                victim = self.running[-1]
                self.preempt(victim)
                if victim is request:
                    break
            if request in self.running:
                scheduled.append((request, num_new))
        while self.waiting:
            request = self.waiting[0]
            hit_blocks, num_hit, _ = self.manager.get_computed_blocks(request)
            # Bounded replay recomputes the hit's last `replay_tokens` tokens
            # and drops a hit no longer than that.
            if num_hit <= self.replay_tokens:
                hit_blocks, num_hit = self.manager.empty_kv_cache_blocks, 0
            num_replay = min(self.replay_tokens, num_hit)
            num_new = min(CHUNK, request.num_tokens - num_hit + num_replay)
            blocks = self.manager.allocate_slots(
                request,
                max(num_new - num_replay, 0),
                num_new_computed_tokens=num_hit,
                new_computed_blocks=hit_blocks,
                has_scheduled_reqs=bool(scheduled),
            )
            self.record(request, blocks, num_hit)
            if blocks is None:
                break
            self.waiting.pop(0)
            request.status = RequestStatus.RUNNING
            request.num_computed_tokens = num_hit - num_replay
            self.running.append(request)
            scheduled.append((request, num_new))
        for request, num_new in scheduled:
            request.num_computed_tokens += num_new
            request.num_in_flight_tokens += num_new
            samples = request.num_computed_tokens >= request.num_tokens
            request.num_output_placeholders += samples
        self.in_flight.append(scheduled)

    def finish_steps(self, lag):
        while len(self.in_flight) > lag:
            for request, num_new in self.in_flight.pop(0):
                request.num_in_flight_tokens -= num_new
                if request.status != RequestStatus.RUNNING:
                    continue
                if request.num_output_placeholders:
                    request.num_output_placeholders -= 1
                    request.append_output_token_ids((request.num_tokens * 7919) % 1000)
                if request.num_output_tokens >= request.max_tokens:
                    self.running.remove(request)
                    request.status = RequestStatus.FINISHED_LENGTH_CAPPED
                    self.manager.free(request)
                    self.record(request, "freed")

    def preempt(self, request):
        self.running.remove(request)
        # The scheduler's deferred free: pop now, return to the pool later.
        blocks = self.manager.pop_blocks_for_free(request)
        self.manager.block_pool.free_blocks(reversed(blocks))
        # Drop the request's scheduled and in-flight tokens, which vLLM
        # discards as stale.
        self.scheduled[:] = [
            entry for entry in self.scheduled if entry[0] is not request
        ]
        for step in self.in_flight:
            step[:] = [entry for entry in step if entry[0] is not request]
        request.status = RequestStatus.PREEMPTED
        request.num_computed_tokens = 0
        request.num_in_flight_tokens = 0
        request.num_output_placeholders = 0
        self.waiting.insert(0, request)
        self.record(request, "preempted")

    def record(self, request, blocks, num_hit=None):
        if blocks is None or isinstance(blocks, str):
            returned = blocks
        else:
            returned = blocks.get_block_ids()
        tables = tuple(
            tuple(block.block_id for block in group.req_to_blocks.get(r.request_id, ()))
            for r in self.running
            for group in self.manager.coordinator.single_type_managers
        )
        self.log.append(
            (
                self.step,
                request.request_id,
                num_hit,
                returned,
                tables,
                self.manager.block_pool.get_num_free_blocks(),
            )
        )


def count_skips(manager):
    """Count calls that reach the coordinator, to show the fast path skips."""
    coordinator = manager.coordinator
    calls = [0]
    remove = coordinator.remove_skipped_blocks

    def counted(*args, **kwargs):
        calls[0] += 1
        return remove(*args, **kwargs)

    coordinator.remove_skipped_blocks = counted
    return calls


@pytest.mark.parametrize("lag", [0, 2])
@pytest.mark.parametrize(
    "groups, enable_caching",
    [
        (sliding_window_groups, True),
        (sliding_window_groups, False),
        (bounded_replay_groups, False),
        (bounded_replay_groups, True),
    ],
)
@pytest.mark.parametrize("num_blocks", [150, 4000])
def test_fast_path_matches_stock_manager(groups, enable_caching, lag, num_blocks):
    requests = make_requests(seed=num_blocks + lag, num_requests=12)
    logs, calls = {}, {}
    for manager_cls in (StockKVCacheManager, FastKVCacheManager, CheckedKVCacheManager):
        manager = make_manager(manager_cls, groups(), num_blocks, enable_caching)
        calls[manager_cls] = count_skips(manager)
        logs[manager_cls] = Scheduler(manager, requests, lag).run()
    expected = logs[StockKVCacheManager]
    assert logs[FastKVCacheManager] == expected
    assert logs[CheckedKVCacheManager] == expected
    # Most decode calls skip the coordinator.
    assert calls[FastKVCacheManager][0] < calls[StockKVCacheManager][0] / 2
    assert calls[CheckedKVCacheManager][0] == calls[StockKVCacheManager][0]
    if num_blocks == 150:
        assert any(entry[3] == "preempted" for entry in expected)


def test_unsupported_manager_keeps_stock_behavior():
    specs = [
        attention(FullAttentionSpec, 32),
        attention(ChunkedLocalAttentionSpec, 32, attention_chunk_size=64),
    ]
    requests = make_requests(seed=0, num_requests=4)
    stock = Scheduler(make_manager(StockKVCacheManager, specs, 2000, True), requests, 1)
    fast_manager = make_manager(FastKVCacheManager, specs, 2000, True)
    calls = count_skips(fast_manager)
    fast = Scheduler(fast_manager, requests, 1)
    assert fast.run() == stock.run()
    assert not fast_manager._kolibri_fast_path_supported
    assert calls[0] > 0


def test_install_is_idempotent_and_can_be_disabled(monkeypatch):
    class Manager(StockKVCacheManager):
        pass

    assert not install_decode_fast_path(Manager, mode="0")
    assert Manager.allocate_slots is StockKVCacheManager.allocate_slots
    assert install_decode_fast_path(Manager, mode="1")
    wrapped = Manager.allocate_slots
    assert install_decode_fast_path(Manager, mode="1")
    assert Manager.allocate_slots is wrapped
    with pytest.raises(ValueError):
        install_decode_fast_path(Manager, mode="yes")


def test_install_skips_other_vllm_versions(monkeypatch):
    class Manager(StockKVCacheManager):
        pass

    monkeypatch.setattr(vllm, "__version__", "0.34.0")
    assert not install_decode_fast_path(Manager, mode="1")
    assert "allocate_slots" not in vars(Manager)


def test_install_skips_other_allocate_slots_signatures():
    class Manager(StockKVCacheManager):
        def allocate_slots(self, request, num_new_tokens, new_argument=None):
            pass

    assert not install_decode_fast_path(Manager, mode="1")
    assert "_kolibri_fast_path_mode" not in vars(Manager)
