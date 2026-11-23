# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""Run vLLM's repetition detection on every n-th generated token only.

Kolibri models loop by repeating passages of several hundred to ~2k tokens
verbatim, so catching them needs `repetition_detection.max_pattern_size`
around 2048. vLLM scans the tail of every sequence on every generated token,
in Python, at a cost linear in the pattern size: at 2048 that halved the
decode step rate of a full batch (measured on Kolibri 1).

The scan only asks whether the tail is currently repeating, so checking every
`stride` tokens gives the same verdict at most `stride - 1` tokens later, at
1/stride of the cost.

XXX: remove once vLLM no longer scans every token. Candidates upstream:
vllm-project/vllm#41539 (rolling-hash detection), vllm-project/vllm#46700
(bounding the scan). A check interval on `RepetitionDetectionParams` is the
smallest fix and worth proposing there; until then this patches
`vllm.v1.core.sched.utils.check_sequence_repetition`, which `check_stop`
calls via its module global.
"""

from collections.abc import Callable, Sequence
from typing import Any

from aleph_alpha_inference.env import env_int

STRIDE_ENV_VAR = "KOLIBRI_VLLM_STRIDE"
DEFAULT_STRIDE = 32

Check = Callable[[Sequence[int], Any], bool]


def strided(check: Check, stride: int) -> Check:
    """`check`, evaluated only when the output length is a multiple of `stride`."""
    if stride <= 1:
        return check

    def strided_check(token_ids: Sequence[int], params: Any) -> bool:
        if len(token_ids) % stride:
            return False
        return check(token_ids, params)

    strided_check.__wrapped__ = check  # type: ignore[attr-defined]
    return strided_check


def stride_from_env() -> int:
    """The stride from `KOLIBRI_VLLM_STRIDE`, or the default if it is unset
    or not an integer."""
    return env_int(STRIDE_ENV_VAR, DEFAULT_STRIDE)


def patch_vllm(stride: int) -> None:
    """Replace vLLM's per-token repetition scan with the strided one. Idempotent."""
    import vllm.v1.core.sched.utils as sched_utils

    current = sched_utils.check_sequence_repetition
    original = getattr(current, "__wrapped__", current)
    sched_utils.check_sequence_repetition = strided(original, stride)
