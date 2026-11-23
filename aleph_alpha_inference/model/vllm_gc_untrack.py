# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""Keep vLLM's per-request lists that grow with the context out of Python's
cyclic GC.

A collection visits every element of every tracked list in the generations it
collects. A 1M-token request brings two prompt lists of a million ints into
the engine core, plus a block list per KV cache group, so one young collection
took 287-521 ms with 32 Kolibri 1.1 requests of 1M per engine on B200. Under
data parallelism every engine waits for the slowest at each step, so such a
pause stalls every GPU. Reference counting still frees these lists, and none
of them can be part of a reference cycle once blocks refer to their pool
weakly.
"""

# XXX: Backport of vllm-project/vllm#65300, merged after vLLM 0.33.0. Remove
# once the plugin requires vLLM 0.34.0, which ships it.

import ctypes
import functools
import gc
import weakref
from collections import defaultdict
from collections.abc import Iterable

from vllm.logger import init_logger

from aleph_alpha_inference.env import env_choice

logger = init_logger(f"vllm.{__name__}")

# A PyDLL call, so it holds the GIL. CPython never re-tracks an untracked list.
_PyObject_GC_UnTrack = ctypes.pythonapi.PyObject_GC_UnTrack
_PyObject_GC_UnTrack.argtypes = [ctypes.py_object]
_PyObject_GC_UnTrack.restype = None

# "1" (default) patches vLLM, "0" leaves its lists tracked.
MODE_ENV = "KOLIBRI_VLLM_GC_UNTRACK"
MODES = ("0", "1")
FIRST_FIXED_VLLM = "0.34.0"


def gc_untrack(obj):
    """Hide a container from the cyclic GC, so that collections skip its
    elements. Reference counting still frees it.

    Only for containers that can never be part of a reference cycle: a cycle
    through an untracked container is never collected. Dicts re-track
    themselves when a tracked value is inserted, so this suits lists.
    A no-op for objects the GC does not track.
    """
    if gc.is_tracked(obj):
        _PyObject_GC_UnTrack(obj)
    return obj


def untracked_list() -> list:
    """An empty list that the cyclic GC never traverses (see `gc_untrack`)."""
    return gc_untrack([])


def untrack_long_lists(lists: Iterable[list | None], min_len: int = 256) -> None:
    """`gc_untrack` the lists with at least `min_len` elements, skipping None.

    Untracking costs ~0.2 us a call, a young collection ~20 ns per element it
    scans, so it only pays off for long lists.
    """
    for values in lists:
        if values is not None and len(values) >= min_len:
            gc_untrack(values)


class BlockHashToBlockMap:
    """vLLM 0.33's prefix-cache index from block hash to cached blocks, holding
    block ids instead of blocks.

    The cyclic GC doesn't track a dict of only bytes and ints, so cached blocks
    add nothing to a collection. Keys cached by several blocks keep all their
    block ids, in insertion order, in `_dup_ids`, so that `_cache` holds no
    containers.
    """

    def __init__(self, blocks):
        self._blocks = blocks
        # Untracked explicitly: Python 3.14+ no longer untracks dicts of atomic
        # values by itself.
        self._cache = gc_untrack({})
        self._dup_ids = {}

    def get_one_block(self, key):
        block_id = self._cache.get(key)
        return None if block_id is None else self._blocks[block_id]

    def contain(self, key, block_id: int) -> bool:
        dup_ids = self._dup_ids.get(key)
        if dup_ids is not None:
            return block_id in dup_ids
        return self._cache.get(key) == block_id

    def insert(self, key, block) -> None:
        block_id = self._cache.setdefault(key, block.block_id)
        if block_id != block.block_id:
            self._dup_ids.setdefault(key, {block_id: None})[block.block_id] = None

    def pop(self, key, block_id: int):
        dup_ids = self._dup_ids.get(key)
        if dup_ids is None:
            if self._cache.get(key) != block_id:
                return None
            del self._cache[key]
        else:
            if dup_ids.pop(block_id, False) is False:
                return None
            self._cache[key] = next(iter(dup_ids))
            if len(dup_ids) == 1:
                del self._dup_ids[key]
        return self._blocks[block_id]

    def __len__(self) -> int:
        return len(self._cache)

    def clear(self) -> None:
        self._cache.clear()
        self._dup_ids.clear()


def _after(cls, name: str, then) -> None:
    """Run `then(self)` after `cls.name(self, ...)`."""
    original = getattr(cls, name)

    @functools.wraps(original)
    def patched(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        then(self)
        return result

    setattr(cls, name, patched)


def _patch_request() -> None:
    from vllm.v1.request import Request

    def untrack(self) -> None:
        # Token ids hold only ints and grow with the context; they are only
        # ever mutated in place. Block hashes hold only bytes.
        for values in (
            self.prompt_token_ids,
            self._all_token_ids,
            self._output_token_ids,
            self.block_hashes,
        ):
            gc_untrack(values)

    def prefill_token_ids(self) -> list[int]:
        """All token ids, as the prompt_token_ids object itself while they
        hold the same tokens, so that pickling both sends them once."""
        prompt = self.prompt_token_ids
        if (
            prompt is not None
            and self.prompt_is_token_ids is None
            and len(prompt) == len(self._all_token_ids)
        ):
            return prompt
        return self._all_token_ids

    _after(Request, "__init__", untrack)
    Request.prefill_token_ids = property(prefill_token_ids)


def _patch_new_request_data() -> None:
    from vllm.v1.core.sched.output import NewRequestData

    from_request = NewRequestData.__dict__["from_request"].__func__

    def from_request_sent_once(
        cls, request, block_ids, prefill_token_ids=None, uses_mrope=False
    ):
        # vLLM 0.34 passes `request.prefill_token_ids` where 0.33's scheduler
        # passes `request._all_token_ids`.
        if prefill_token_ids is request._all_token_ids:
            prefill_token_ids = request.prefill_token_ids
        return from_request(cls, request, block_ids, prefill_token_ids, uses_mrope)

    def setstate(self, state: dict) -> None:
        self.__dict__.update(state)
        # Workers receive these by pickle and keep them only for this step.
        untrack_long_lists(
            (
                self.prompt_token_ids,
                self.prefill_token_ids,
                self.prompt_is_token_ids,
                *self.block_ids,
            )
        )

    NewRequestData.from_request = classmethod(from_request_sent_once)
    NewRequestData.__setstate__ = setstate


def _patch_kv_cache_managers() -> None:
    from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager

    def untrack(self) -> None:
        # Blocks refer to their pool only weakly (`_patch_block_pool`), so
        # these lists cannot be part of a cycle.
        self.req_to_blocks = defaultdict(untracked_list)

    _after(SingleTypeKVCacheManager, "__init__", untrack)


def _free_blocks(self, ordered_blocks) -> None:
    """vLLM 0.33's `BlockPool.free_blocks`, for blocks that refer to their
    pool weakly."""
    blocks_to_evict_last = []
    blocks_to_evict_first = []
    other_pools = {}
    for block in ordered_blocks:
        if block.is_null:  # never pooled; its ref_cnt is not maintained
            continue
        if block.pool is not None and block.pool is not self._weak_self:
            pool = block.pool()
            if pool is not None:
                other_pools.setdefault(pool, []).append(block)
            continue
        block.ref_cnt -= 1
        if block.ref_cnt == 0:
            if block.block_hash is None or not self.enable_caching:
                # LIFO reuse of non-cached blocks for better GPU locality.
                blocks_to_evict_first.append(block)
            else:
                # FIFO reuse of cached blocks for LRU eviction behavior.
                blocks_to_evict_last.append(block)
    self.free_block_queue.prepend_n(blocks_to_evict_first)
    self.free_block_queue.append_n(blocks_to_evict_last)
    for pool, blocks in other_pools.items():
        pool.free_blocks(blocks)


def _patch_block_pool() -> None:
    from vllm.v1.core.block_pool import BlockPool

    def weak_blocks(self) -> None:
        # Weak, so that block -> pool cannot close a cycle through the
        # untracked block lists.
        self._weak_self = weakref.ref(self)
        for block in self.blocks:
            block.pool = self._weak_self
        self.cached_block_hash_to_block = BlockHashToBlockMap(self.blocks)

    reset_prefix_cache = BlockPool.reset_prefix_cache

    @functools.wraps(reset_prefix_cache)
    def reset_prefix_cache_in_place(self) -> bool:
        # vLLM 0.33 replaces the index with its own kind; keep ours, cleared.
        cache = self.cached_block_hash_to_block
        reset = reset_prefix_cache(self)
        if reset:
            cache.clear()
            self.cached_block_hash_to_block = cache
        return reset

    _after(BlockPool, "__init__", weak_blocks)
    BlockPool.free_blocks = _free_blocks
    BlockPool.reset_prefix_cache = reset_prefix_cache_in_place


def _patch_front_end() -> None:
    from vllm.entrypoints.openai.responses.context import ParsableContext, SimpleContext
    from vllm.v1.engine.detokenizer import (
        IncrementalDetokenizer,
        SlowIncrementalDetokenizer,
    )
    from vllm.v1.engine.output_processor import RequestState

    def untrack_prompt(self) -> None:
        # The same list as every other holder of this prompt, and only extended
        # in place, so hiding it here covers them all.
        gc_untrack(self.prompt_token_ids)

    _after(RequestState, "__init__", untrack_prompt)
    _after(RequestState, "apply_streaming_update", untrack_prompt)
    # The output token ids, and one string per output token, grow in place.
    _after(IncrementalDetokenizer, "__init__", lambda self: gc_untrack(self.token_ids))
    _after(SlowIncrementalDetokenizer, "__init__", lambda self: gc_untrack(self.tokens))
    for context in (SimpleContext, ParsableContext):
        _after(
            context,
            "__init__",
            lambda self: gc_untrack(self._accumulated_token_ids),
        )


def untrack_request_lists(mode: str | None = None) -> bool:
    """Patch vLLM to keep its per-request token, block hash and block lists
    out of the cyclic GC, in the engine core, the GPU workers and the API
    servers.

    Args:
        mode: One of `MODES`; read from `KOLIBRI_VLLM_GC_UNTRACK` when None,
            defaulting to "1", also for a value that is not a mode. "0" only keeps the plugin from patching: a
            vLLM that ships the fix untracks these lists regardless.

    Returns:
        Whether vLLM was patched: not when switched off, when it ships the
        fix or when it was patched already.
    """
    import vllm
    from packaging.version import Version
    from vllm.v1.request import Request

    if mode is None:
        mode = env_choice(MODE_ENV, MODES, "1")
    if mode not in MODES:
        raise ValueError(f"{MODE_ENV} must be one of {MODES}, got {mode!r}")
    if mode == "0":
        logger.info("Kolibri leaves vLLM's request lists in the GC: %s=0", MODE_ENV)
        return False
    if Version(vllm.__version__) >= Version(FIRST_FIXED_VLLM):
        logger.info("vLLM %s keeps request lists out of the GC", vllm.__version__)
        return False
    if hasattr(Request, "prefill_token_ids"):
        return False
    _patch_request()
    _patch_new_request_data()
    _patch_kv_cache_managers()
    _patch_block_pool()
    _patch_front_end()
    logger.info(
        "Kolibri keeps vLLM's per-request lists out of the cyclic GC (%s=0 "
        "leaves them tracked)",
        MODE_ENV,
    )
    return True
