# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""The full-GC relaxation changes only the full-collection threshold."""

import gc

import pytest
from aleph_alpha_inference.model.vllm_gc import (
    FULL_GC_THRESHOLD,
    MODE_ENV,
    relax_full_gc,
)


@pytest.fixture(autouse=True)
def restore_threshold():
    threshold = gc.get_threshold()
    gc.set_threshold(700, 10, 10)
    yield
    gc.set_threshold(*threshold)


@pytest.fixture
def initial_threshold() -> tuple[int, int, int]:
    """The thresholds the tests start from. Python 3.14 releases with the
    incremental collector ignore the full-collection threshold and report it
    as 0."""
    return gc.get_threshold()


def test_relax_full_gc_raises_only_the_full_threshold(monkeypatch, initial_threshold):
    monkeypatch.delenv(MODE_ENV, raising=False)
    has_full_threshold = initial_threshold == (700, 10, 10)
    assert relax_full_gc() is has_full_threshold
    if has_full_threshold:
        assert gc.get_threshold() == (700, 10, FULL_GC_THRESHOLD)
    else:
        assert gc.get_threshold() == initial_threshold


def test_relax_full_gc_can_be_disabled(monkeypatch, initial_threshold):
    monkeypatch.setenv(MODE_ENV, "0")
    assert not relax_full_gc()
    assert gc.get_threshold() == initial_threshold


def test_bad_switch_value_relaxes_as_by_default(monkeypatch, initial_threshold):
    monkeypatch.setenv(MODE_ENV, "yes")
    assert relax_full_gc() is (initial_threshold == (700, 10, 10))
