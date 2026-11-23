# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""Variable eval lengths must reuse executable code without changing outputs."""

import pytest
import torch

pytestmark = pytest.mark.gpu


def _static_pack(op, tensors):
    import cuda.bindings.driver as cuda
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    args = [
        from_dlpack(t.detach(), assumed_align=2 if t.dtype == torch.bfloat16 else 4)
        for t in tensors
    ]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    cute.compile(op, *args, stream)(*args, stream)


def test_query_pack_reuses_code_across_token_counts():
    from aleph_alpha_inference.kernels import decode_pack as pack

    pack._COMPILED.clear()
    torch.manual_seed(734)
    for rows in (1, 17, 65, 257, 513):
        q = torch.randn(rows, 48, 128, device="cuda", dtype=torch.bfloat16)

        def buffers():
            return (
                torch.zeros(rows, 16, 64, device="cuda", dtype=torch.int8),
                torch.zeros(rows, 16, 2, device="cuda", dtype=torch.int32),
            )

        expected, actual = buffers(), buffers()
        _static_pack(
            pack.PackQuery(),
            (
                q,
                expected[0].view(torch.uint8),
                expected[1].view(torch.float8_e4m3fn),
            ),
        )
        pack.DecodeQueryPack()(q, *actual)
        for got, ref in zip(actual, expected):
            torch.testing.assert_close(got, ref, rtol=0, atol=0)
        assert len(pack._COMPILED) == 1


def test_compressed_selection_reuses_code_across_prefill_rows():
    from aleph_alpha_inference.kernels import serving_compressed as routing
    from aleph_alpha_inference.kernels.decode_selection import RadixSelection

    routing._COMPILED.clear()
    torch.manual_seed(736)
    for rows in (1, 3, 17, 65, 129):
        scores = torch.rand(rows, 256, device="cuda")
        lengths = torch.full((rows,), 4096, device="cuda", dtype=torch.int32)
        selected = torch.zeros(rows, 16, device="cuda", dtype=torch.int32)
        routing._COMPILED.launch(
            RadixSelection(0, threads=128, topk=16, block_size=16),
            (scores, selected, lengths, selected, selected, lengths),
            key=(16, 16, False),
        )
        torch.testing.assert_close(
            selected.sort(dim=1).values,
            scores.topk(16, dim=1).indices.int().sort(dim=1).values,
            rtol=0,
            atol=0,
        )
        assert len(routing._COMPILED) == 1
