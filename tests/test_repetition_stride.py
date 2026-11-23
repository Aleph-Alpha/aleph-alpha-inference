# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
import vllm.v1.core.sched.utils as sched_utils
from vllm.sampling_params import RepetitionDetectionParams
from aleph_alpha_inference import register
from aleph_alpha_inference.repetition import (
    DEFAULT_STRIDE,
    STRIDE_ENV_VAR,
    stride_from_env,
    strided,
)


def _repeats(token_ids, params) -> bool:
    """Stand-in for vLLM's scan: the tail repeats the pattern `params` times."""
    pattern = [1, 2, 3]
    tail = pattern * params
    return list(token_ids[-len(tail) :]) == tail


def test_strided_check_skips_off_stride_lengths():
    check = strided(_repeats, stride=32)
    looping = [0] * 5 + [1, 2, 3] * 20  # 65 tokens, repeating since token 5

    assert _repeats(looping, 8)
    assert not check(looping, 8)


def test_strided_check_delegates_on_stride_lengths():
    check = strided(_repeats, stride=32)
    looping = [0] * 8 + [1, 2, 3] * 40  # 128 tokens

    assert check(looping, 8)
    assert not check([0] * 128, 8)


def test_stride_of_one_keeps_the_original_check():
    assert strided(_repeats, stride=1) is _repeats


def test_stride_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv(STRIDE_ENV_VAR, "8")

    assert stride_from_env() == 8


def test_bad_stride_falls_back_to_the_default(monkeypatch):
    monkeypatch.setenv(STRIDE_ENV_VAR, "often")

    assert stride_from_env() == DEFAULT_STRIDE


def test_register_patches_vllm_idempotently(register_plugin):
    patched = sched_utils.check_sequence_repetition
    assert patched.__name__ == "strided_check"
    original = patched.__wrapped__

    register()

    assert sched_utils.check_sequence_repetition.__wrapped__ is original


def test_patched_vllm_scan_fires_on_a_stride_multiple(register_plugin):
    params = RepetitionDetectionParams(
        max_pattern_size=8, min_pattern_size=1, min_count=4
    )
    looping = [7] * 3 + [1, 2] * 100

    assert not sched_utils.check_sequence_repetition(looping[:193], params)
    assert sched_utils.check_sequence_repetition(looping[:192], params)
