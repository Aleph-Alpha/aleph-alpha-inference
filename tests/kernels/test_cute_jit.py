# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""Shape-agnostic TVM-FFI executables load from FA4's persistent cache."""

import pytest
import torch

pytestmark = pytest.mark.gpu


def _pack(executables, q):
    """Pack queries `q` with `executables`."""
    from aleph_alpha_inference.kernels.decode_pack import PackQuery

    rows = q.shape[0]
    packed = torch.zeros(rows, 16, 64, device="cuda", dtype=torch.uint8)
    scales = torch.zeros(rows, 16, 2, device="cuda", dtype=torch.int32)
    tensors = (q, packed, scales.view(torch.float8_e4m3fn))
    executables.launch(
        PackQuery(),
        tensors,
        aligns=[2 if t.dtype == torch.bfloat16 else 4 for t in tensors],
    )
    return packed, scales


def test_executables_serve_every_row_count_and_persist(monkeypatch, tmp_path):
    import cutlass.cute as cute

    from aleph_alpha_inference.kernels import cute_jit

    monkeypatch.setattr(cute_jit.cache_utils, "CUTE_DSL_CACHE_ENABLED", True)
    monkeypatch.setattr(cute_jit.cache_utils, "get_cache_path", lambda: tmp_path)
    compiles = []
    native = cute.compile
    monkeypatch.setattr(
        cute, "compile", lambda *a, **k: compiles.append(1) or native(*a, **k)
    )
    torch.manual_seed(21)
    queries = {
        rows: torch.randn(rows, 48, 128, device="cuda", dtype=torch.bfloat16)
        for rows in (1, 7, 129)
    }
    first = cute_jit.Executables("test-query-pack")
    expected = {rows: _pack(first, q) for rows, q in queries.items()}
    assert len(first) == 1 and len(compiles) == 1
    # A restarted process loads the exported executable instead of compiling.
    del cute_jit.CACHES["test-query-pack"]
    second = cute_jit.Executables("test-query-pack")
    for got, want in zip(_pack(second, queries[7]), expected[7]):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    assert len(compiles) == 1
    del cute_jit.CACHES["test-query-pack"]
