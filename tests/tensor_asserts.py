# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""Tensor assertions shared by the plugin's tests."""

import torch


def assert_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Bit-wise equality of two tensors, reporting the differing elements."""
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
