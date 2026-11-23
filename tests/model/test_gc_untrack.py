# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""Backport of vllm-project/vllm#65300: vLLM's per-request lists stay out of
the cyclic GC."""

import copy
import gc
import weakref
from array import array

import pytest
from vllm.sampling_params import SamplingParams
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHashWithGroupId, KVCacheBlock
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest
from vllm.v1.engine.output_processor import OutputProcessor
from vllm.v1.request import Request
from aleph_alpha_inference.model import vllm_gc_untrack
from aleph_alpha_inference.model.vllm_gc_untrack import (
    MODE_ENV,
    BlockHashToBlockMap,
    gc_untrack,
    untrack_request_lists,
    untracked_list,
)


@pytest.fixture(scope="module", autouse=True)
def patched_vllm():
    untrack_request_lists("1")


def test_untracking_can_be_disabled(monkeypatch):
    def fail():
        raise AssertionError("vLLM was patched")

    # As in a process the plugin has not patched yet.
    monkeypatch.delattr(Request, "prefill_token_ids")
    monkeypatch.setattr(vllm_gc_untrack, "_patch_request", fail)
    monkeypatch.setenv(MODE_ENV, "0")
    assert not untrack_request_lists()
    with pytest.raises(ValueError):
        untrack_request_lists("yes")


def test_bad_switch_value_patches_as_by_default(monkeypatch):
    patched = []
    monkeypatch.delattr(Request, "prefill_token_ids")
    for name in (
        "_patch_request",
        "_patch_new_request_data",
        "_patch_kv_cache_managers",
        "_patch_block_pool",
        "_patch_front_end",
    ):
        monkeypatch.setattr(
            vllm_gc_untrack, name, lambda name=name: patched.append(name)
        )
    monkeypatch.setenv(MODE_ENV, "yes")
    assert untrack_request_lists()
    assert len(patched) == 5


def long_request() -> Request:
    return Request("r", list(range(1000)), SamplingParams(max_tokens=4), None)


def test_untracked_list_stays_untracked():
    blocks = untracked_list()
    blocks.extend(KVCacheBlock(i) for i in range(4))
    blocks.append(KVCacheBlock(4))
    blocks[0] = KVCacheBlock(5)
    del blocks[-2:]
    assert not gc.is_tracked(blocks)


def test_gc_untrack_ignores_untracked_objects():
    ids = array("i", range(4))
    assert gc_untrack(ids) is ids and gc_untrack(None) is None


def test_request_token_lists_are_untracked():
    request = long_request()
    assert not gc.is_tracked(request.prompt_token_ids)
    assert not gc.is_tracked(request._all_token_ids)
    assert not gc.is_tracked(request._output_token_ids)


def test_new_request_carries_the_prompt_once():
    request = long_request()
    data = NewRequestData.from_request(request, ([1],), request._all_token_ids)
    assert data.prefill_token_ids is data.prompt_token_ids


def test_prefill_token_ids_leave_the_prompt_once_outputs_exist():
    request = long_request()
    request.append_output_token_ids(7)
    assert request.prefill_token_ids is request._all_token_ids


def test_unpickled_long_lists_are_untracked():
    long, short = list(range(1000)), [1, 2]
    request = Request("r", long, SamplingParams(max_tokens=4), None)
    data = NewRequestData.from_request(request, (long, short), long)
    received = copy.deepcopy(data)  # the same __setstate__ path as pickle
    assert not gc.is_tracked(received.prompt_token_ids)
    assert not gc.is_tracked(received.prefill_token_ids)
    assert not gc.is_tracked(received.block_ids[0])
    assert gc.is_tracked(received.block_ids[1])
    assert received.prompt_token_ids == long


def test_block_lookup_cache_keeps_every_block_of_a_key():
    blocks = [KVCacheBlock(i) for i in range(4)]
    cache = BlockHashToBlockMap(blocks)
    key0, key1 = BlockHashWithGroupId(b"hash0"), BlockHashWithGroupId(b"hash1")
    cache.insert(key0, blocks[0])
    cache.insert(key0, blocks[3])
    cache.insert(key1, blocks[1])
    assert not gc.is_tracked(cache._cache)
    assert cache._cache == {key0: 0, key1: 1}
    assert cache._dup_ids == {key0: {0: None, 3: None}}
    assert cache.contain(key0, 3) and not cache.contain(key1, 3)
    assert cache.pop(key0, 0) is blocks[0]
    assert cache.get_one_block(key0) is blocks[3]
    assert cache._dup_ids == {}
    assert cache.pop(key0, 0) is None
    assert len(cache) == 2


def test_blocks_refer_to_their_pool_weakly():
    pool = BlockPool(num_gpu_blocks=4, enable_caching=True, hash_block_size=16)
    blocks = pool.get_new_blocks(2)
    assert all(isinstance(block.pool, weakref.ref) for block in pool.blocks)
    pool.free_blocks(blocks)
    assert pool.get_num_free_blocks() == 3


def test_freeing_a_dropped_pool_s_blocks_is_skipped():
    other = BlockPool(num_gpu_blocks=4, enable_caching=True, hash_block_size=16)
    pool = BlockPool(num_gpu_blocks=4, enable_caching=True, hash_block_size=16)
    blocks = other.get_new_blocks(2)
    del other
    gc.collect()
    pool.free_blocks(blocks)
    assert pool.get_num_free_blocks() == 3


def test_reset_prefix_cache_clears_the_index_in_place():
    pool = BlockPool(num_gpu_blocks=4, enable_caching=True, hash_block_size=16)
    cache = pool.cached_block_hash_to_block
    cache.insert(BlockHashWithGroupId(b"hash0"), pool.blocks[1])
    assert pool.reset_prefix_cache()
    assert pool.cached_block_hash_to_block is cache
    assert not cache and not gc.is_tracked(cache._cache)


def test_front_end_request_lists_are_untracked():
    output_processor = OutputProcessor(tokenizer=None, log_stats=False)
    prompt_tokens = [1, 2, 3, 4]
    request = EngineCoreRequest(
        request_id="request-0-int",
        external_req_id="request-0",
        prompt_token_ids=prompt_tokens,
        mm_features=None,
        arrival_time=0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        sampling_params=SamplingParams(detokenize=False),
        pooling_params=None,
    )
    output_processor.add_request(request, prompt=None)
    state = output_processor.request_states["request-0-int"]
    output_processor.process_outputs(
        [EngineCoreOutput(request_id="request-0-int", new_token_ids=[42])]
    )
    assert state.prompt_token_ids is prompt_tokens
    assert not gc.is_tracked(prompt_tokens)
    assert state.detokenizer.token_ids == [42]
    assert not gc.is_tracked(state.detokenizer.token_ids)
