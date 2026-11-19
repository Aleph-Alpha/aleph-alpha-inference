# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""Decode fast path for vLLM's `KVCacheManager.allocate_slots`.

In steady decode, a request needs a new KV block, frees a sliding-window block
or fills a prefix-cache block only when it crosses a block boundary of some
KV-cache group. vLLM still walks every group for every request on every step.
Kolibri has 21 groups, so this bookkeeping dominates the engine-core host time
in decode. The fast path returns early from plain decode calls until a request
reaches its next boundary.

XXX: this belongs in vLLM upstream. Remove it once vLLM skips these calls
itself. It is installed only on the vLLM minor version it was checked against.
"""

import functools
import inspect
import math

import vllm
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    SlidingWindowManager,
)
from vllm.v1.request import RequestStatus

from aleph_alpha_inference.env import env_choice

logger = init_logger(f"vllm.{__name__}")

# "1" (default) skips calls, "0" turns the fast path off, "check" runs the
# real call anyway and asserts that skipping it would have changed nothing.
MODE_ENV = "KOLIBRI_VLLM_DECODE_FAST_PATH"
MODES = ("0", "1", "check")
SUPPORTED_VLLM_VERSION = "0.33."
# The wrapper passes these on by position, so it must match vLLM's order.
ALLOCATE_SLOTS_PARAMETERS = (
    "self",
    "request",
    "num_new_tokens",
    "num_new_computed_tokens",
    "new_computed_blocks",
    "num_lookahead_tokens",
    "num_external_computed_tokens",
    "delay_cache_blocks",
    "num_encoder_tokens",
    "full_sequence_must_fit",
    "reserved_blocks",
    "has_scheduled_reqs",
    "skip_zeroing_group_ids",
)


def install_decode_fast_path(manager_cls=KVCacheManager, mode=None) -> bool:
    """Wrap `allocate_slots` and the free methods of a KV-cache manager class.

    Args:
        manager_cls: `KVCacheManager` or a subclass, patched in place.
        mode: One of `MODES`; read from `KOLIBRI_VLLM_DECODE_FAST_PATH` when
            None, defaulting to "1", also for a value that is not a mode.

    Returns:
        Whether the class carries the fast path after the call. Installing
        twice is a no-op. Skipped with a log line on other vLLM versions and
        when `allocate_slots` takes other parameters.
    """
    if mode is None:
        mode = env_choice(MODE_ENV, MODES, "1")
    if mode not in MODES:
        raise ValueError(f"{MODE_ENV} must be one of {MODES}, got {mode!r}")
    if "_kolibri_fast_path_mode" in vars(manager_cls):
        return True
    if mode == "0":
        logger.info("Kolibri decode fast path disabled by %s=0", MODE_ENV)
        return False
    if not vllm.__version__.startswith(SUPPORTED_VLLM_VERSION):
        logger.warning(
            "Kolibri decode fast path skipped: written for vLLM %sx, found %s",
            SUPPORTED_VLLM_VERSION,
            vllm.__version__,
        )
        return False
    parameters = tuple(inspect.signature(manager_cls.allocate_slots).parameters)
    if parameters != ALLOCATE_SLOTS_PARAMETERS:
        logger.warning(
            "Kolibri decode fast path skipped: vLLM %s's allocate_slots takes %s",
            vllm.__version__,
            parameters,
        )
        return False

    manager_cls.allocate_slots = _wrap_allocate_slots(manager_cls.allocate_slots, mode)
    manager_cls.free = _wrap_free(manager_cls.free)
    manager_cls.pop_blocks_for_free = _wrap_free(manager_cls.pop_blocks_for_free)
    manager_cls._kolibri_fast_path_mode = mode
    logger.info("Kolibri decode fast path installed (mode %s)", mode)
    return True


def _wrap_allocate_slots(original, mode):
    """Return `allocate_slots` that skips plain decode calls between boundaries."""

    @functools.wraps(original)
    def allocate_slots(
        self,
        request,
        num_new_tokens,
        num_new_computed_tokens=0,
        new_computed_blocks=None,
        num_lookahead_tokens=0,
        num_external_computed_tokens=0,
        delay_cache_blocks=False,
        num_encoder_tokens=0,
        full_sequence_must_fit=False,
        reserved_blocks=0,
        has_scheduled_reqs=True,
        skip_zeroing_group_ids=(),
    ):
        args = (
            request,
            num_new_tokens,
            num_new_computed_tokens,
            new_computed_blocks,
            num_lookahead_tokens,
            num_external_computed_tokens,
            delay_cache_blocks,
            num_encoder_tokens,
            full_sequence_must_fit,
            reserved_blocks,
            has_scheduled_reqs,
            skip_zeroing_group_ids,
        )
        plain_decode = (
            num_new_tokens > 0
            and request.status == RequestStatus.RUNNING
            and num_new_computed_tokens == 0
            and (new_computed_blocks is None or not any(new_computed_blocks.blocks))
            and num_lookahead_tokens == 0
            and num_external_computed_tokens == 0
            and not delay_cache_blocks
            and num_encoder_tokens == 0
            and not full_sequence_must_fit
            and reserved_blocks == 0
            and _supported(self, mode)
        )
        states = self.__dict__.get("_kolibri_fast_path_states")
        if not plain_decode:
            if states is not None:
                states.pop(request.request_id, None)
            return original(self, *args)

        # The same token counts the real call derives.
        total = min(request.num_computed_tokens, self.max_model_len)
        num_tokens_need_slot = min(total + num_new_tokens, self.max_model_len)
        processed = max(0, total - request.num_in_flight_tokens)
        num_tokens_to_cache = min(total + num_new_tokens, request.num_tokens)
        cache_unit = num_tokens_to_cache // self._kolibri_fast_path_boundary
        state = states.get(request.request_id)
        skip = (
            state is not None
            and num_tokens_need_slot <= state[0]
            and processed < state[1]
            and cache_unit == state[2]
        )
        if skip and mode == "1":
            return self.empty_kv_cache_blocks

        before = _snapshot(self, request.request_id) if skip else None
        result = original(self, *args)
        if skip:
            assert (
                result is self.empty_kv_cache_blocks
            ), f"Decode fast path would skip an allocation for {request.request_id}"
            assert (
                _snapshot(self, request.request_id) == before
            ), f"Decode fast path would skip a block-table change for {request.request_id}"
        if result is None:
            states.pop(request.request_id, None)
        else:
            covered, next_removal = _thresholds(self, request.request_id, processed)
            states[request.request_id] = (covered, next_removal, cache_unit)
        return result

    return allocate_slots


def _wrap_free(original):
    """Return a free method that also drops the request's fast-path state."""

    @functools.wraps(original)
    def free(self, request):
        states = self.__dict__.get("_kolibri_fast_path_states")
        if states is not None:
            states.pop(request.request_id, None)
        return original(self, request)

    return free


def _supported(manager, mode) -> bool:
    """Whether every KV-cache group of `manager` changes only at the boundaries
    the fast path tracks. Decided once per manager; also sets up its state."""
    supported = manager.__dict__.get("_kolibri_fast_path_supported")
    if supported is not None:
        return supported
    groups = manager.coordinator.single_type_managers
    # Subclasses of FullAttentionManager (sink, R-SWA, HiSparse) change blocks
    # outside the boundaries below; BoundedReplaySWAManager only changes how a
    # prefix hit is admitted.
    # Multi-module MTP and EAGLE cache a prefix that trails the computed tokens
    # by a few tokens, which the cache boundary below does not track.
    supported = (
        all(
            type(group) is FullAttentionManager
            or isinstance(group, SlidingWindowManager)
            for group in groups
        )
        and not manager.use_eagle
        and manager.coordinator.num_reprefillable_tokens == 0
    )
    names = sorted({type(group).__name__ for group in groups})
    if supported:
        manager._kolibri_fast_path_states = {}
        # The token counts at which any group caches a block are multiples of
        # this, including the partial-tail hash boundary. The hybrid
        # coordinator rounds to the scheduler block size, a common multiple of
        # the group block sizes.
        manager._kolibri_fast_path_boundary = math.gcd(
            manager.block_pool.hash_block_size,
            *(group.block_size for group in groups),
        )
        logger.info(
            "Kolibri decode fast path active (mode %s) for %d KV-cache groups %s",
            mode,
            len(groups),
            names,
        )
    else:
        logger.warning(
            "Kolibri decode fast path inactive for KV-cache managers %s "
            "(eagle %s, reprefillable tokens %d)",
            names,
            manager.use_eagle,
            manager.coordinator.num_reprefillable_tokens,
        )
    manager._kolibri_fast_path_supported = supported
    return supported


def _thresholds(manager, request_id, processed) -> tuple[int, float]:
    """Tokens the request's blocks hold in every group, and the smallest
    processed-token count at which a sliding-window group may free a block
    (infinite without sliding-window groups)."""
    groups = manager.coordinator.single_type_managers
    covered = min(
        len(group.req_to_blocks[request_id]) * group.block_size for group in groups
    )
    next_removal = math.inf
    for group in groups:
        if isinstance(group, SlidingWindowManager):
            # Skipped tokens grow by at most one per processed token, and a
            # block is freed only when they reach the next multiple of the
            # block size.
            skipped = group.get_num_skipped_tokens(processed)
            next_removal = min(
                next_removal, processed + group.block_size - skipped % group.block_size
            )
    return covered, next_removal


def _snapshot(manager, request_id) -> tuple:
    """Block tables, cached-block counts and free blocks a skipped call must
    leave unchanged."""
    return (
        tuple(
            (
                tuple(block.block_id for block in group.req_to_blocks[request_id]),
                group.num_cached_block.get(request_id),
            )
            for group in manager.coordinator.single_type_managers
        ),
        manager.block_pool.get_num_free_blocks(),
    )
