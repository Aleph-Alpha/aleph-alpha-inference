# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""Fresh inputs reuse warmed kernels."""

import pytest
import torch


@pytest.mark.gpu
def test_query_pack_reuses_warmed_shapes_with_fresh_inputs(monkeypatch):
    import cutlass.cute as cute
    from aleph_alpha_inference.kernels.decode_pack import DecodeQueryPack

    def run(rows):
        q = torch.ones(rows, 48, 128, device="cuda", dtype=torch.bfloat16)
        packed = torch.zeros(rows, 16, 64, device="cuda", dtype=torch.int8)
        scales = torch.zeros(rows, 16, 2, device="cuda", dtype=torch.int32)
        DecodeQueryPack()(q, packed, scales)
        return packed, scales

    expected = {rows: run(rows) for rows in (8, 16)}

    def unexpected_compile(*args, **kwargs):
        pytest.fail("Previously warmed query shape compiled again")

    monkeypatch.setattr(cute, "compile", unexpected_compile)
    for rows in (16, 8, 16):
        for actual, reference in zip(run(rows), expected[rows]):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
