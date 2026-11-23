# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""A bad environment variable value warns and gives the default."""

import logging

import pytest
from aleph_alpha_inference import env
from aleph_alpha_inference.env import env_choice, env_int


@pytest.fixture
def warnings(monkeypatch) -> list[str]:
    """The warnings logged by the module. Collected by a handler of its own:
    vLLM's logger does not propagate to pytest's `caplog`."""
    messages: list[str] = []

    class Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            messages.append(record.getMessage())

    handler = Collect(logging.WARNING)
    env.logger.addHandler(handler)
    yield messages
    env.logger.removeHandler(handler)


def test_choice_reads_valid_values_silently(monkeypatch, warnings):
    monkeypatch.delenv("KOLIBRI_TEST", raising=False)
    assert env_choice("KOLIBRI_TEST", ("0", "1"), "1") == "1"
    monkeypatch.setenv("KOLIBRI_TEST", "0")
    assert env_choice("KOLIBRI_TEST", ("0", "1"), "1") == "0"
    assert not warnings


@pytest.mark.parametrize("value", ["yes", "", " 0", "2"])
def test_bad_choice_warns_and_gives_the_default(monkeypatch, warnings, value):
    monkeypatch.setenv("KOLIBRI_TEST", value)
    assert env_choice("KOLIBRI_TEST", ("0", "1"), "1") == "1"
    assert len(warnings) == 1 and "KOLIBRI_TEST" in warnings[0]


def test_int_reads_valid_values_silently(monkeypatch, warnings):
    monkeypatch.delenv("KOLIBRI_TEST", raising=False)
    assert env_int("KOLIBRI_TEST", 32) == 32
    monkeypatch.setenv("KOLIBRI_TEST", "8")
    assert env_int("KOLIBRI_TEST", 32) == 8
    assert not warnings


@pytest.mark.parametrize("value", ["often", "", "1.5"])
def test_bad_int_warns_and_gives_the_default(monkeypatch, warnings, value):
    monkeypatch.setenv("KOLIBRI_TEST", value)
    assert env_int("KOLIBRI_TEST", 32) == 32
    assert len(warnings) == 1 and "KOLIBRI_TEST" in warnings[0]
