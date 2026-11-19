# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
import pytest
import torch


@pytest.mark.gpu
def test_query_pack_rows_past_32_bit_offsets():
    """Q rows are 6,144 elements: past ~349k rows 32-bit offsets overflow."""
    from aleph_alpha_inference.kernels.decode_pack import DecodeQueryPack

    rows = 360000
    q = torch.randn(rows, 1, 48, 128, device="cuda", dtype=torch.bfloat16)
    packed = torch.zeros(rows, 16, 64, device="cuda", dtype=torch.int8)
    scales = torch.zeros(rows, 16, 2, device="cuda", dtype=torch.int32)
    pack = DecodeQueryPack()
    pack(q, packed, scales)
    tail = slice(rows - 4096, rows)
    ref_packed = torch.zeros(4096, 16, 64, device="cuda", dtype=torch.int8)
    ref_scales = torch.zeros(4096, 16, 2, device="cuda", dtype=torch.int32)
    pack(q[tail].contiguous(), ref_packed, ref_scales)
    assert torch.equal(packed[tail], ref_packed)
    assert torch.equal(scales[tail], ref_scales)
