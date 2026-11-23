# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""Checks that the serving image ships its tools and runs the packaged sparse kernels.

They need the image's native build setup (nvcc), so they carry the `image`
marker. CI runs them in the image from a checkout.
"""

import shutil

import pytest

pytestmark = pytest.mark.image

_CACHE_VARIABLES = (
    "TORCH_EXTENSIONS_DIR",
    "CUTE_DSL_CACHE_DIR",
    "CUTLASS_DSL_CACHE_DIR",
    "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "TRITON_CACHE_DIR",
)


@pytest.fixture(autouse=True)
def fresh_caches(tmp_path, monkeypatch):
    # No inherited host caches: these checks must compile from packaged sources.
    for variable in _CACHE_VARIABLES:
        monkeypatch.setenv(variable, str(tmp_path / variable.lower()))


def test_serving_tools_are_installed():
    """nvcc is on PATH for the native kernels."""
    assert shutil.which("nvcc")


@pytest.mark.gpu
def test_packaged_serving_kernels_run():
    """NVFP4/page128 mean16 decode and prefill kernels run for 1k and 2k routes."""
    import torch
    import vllm.vllm_flash_attn  # noqa: F401
    from aleph_alpha_inference.kernels.serving_compressed import (
        route_width,
        routing_workspace,
        ServingCompressedLayer,
        source_pool,
    )
    from aleph_alpha_inference.kernels.sparse_decode import workspace_shapes
    from aleph_alpha_inference.kernels.summary_cache import SummaryCache

    length = 17017
    page = 128
    blocks = (length + page - 1) // page
    torch.manual_seed(714)
    key = torch.randn(blocks * page, 4, 128, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    query = torch.randn(1, 1, 48, 128, device="cuda", dtype=torch.bfloat16)
    table = torch.arange(blocks, device="cuda", dtype=torch.int32)[None]
    lengths = torch.tensor([length], device="cuda", dtype=torch.int32)
    cu = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    q = torch.randn(16, 1, 48, 128, device="cuda", dtype=torch.bfloat16)
    starts = torch.tensor([0, 16], device="cuda", dtype=torch.int32)
    summary_cache = SummaryCache.allocate(blocks, page)
    summary_cache.append(key, value, torch.arange(blocks * page, device="cuda"))
    for topk in (1024, 2048):
        for decode, queries, boundaries in ((True, query, cu), (False, q, starts)):
            rows, owner = queries.shape[0], ("image", topk)
            op = ServingCompressedLayer(
                summary_cache,
                table,
                rows,
                length,
                topk,
                decode,
                scratch=routing_workspace("cuda", owner, rows, length, topk, decode),
                source_pool=source_pool("cuda", owner, decode, rows),
            )
            result = torch.empty_like(queries)
            attention_workspace = decode and tuple(
                torch.empty(shape, device="cuda", dtype=torch.float32)
                for shape in workspace_shapes(rows, op.attention.splits)
            )
            op.run(
                queries,
                boundaries,
                lengths,
                role="source",
                out=result,
                bounds=[],
                query_starts=tuple(boundaries.tolist()),
                attention_workspace=attention_workspace or None,
            )
            assert torch.isfinite(result).all()
            assert (op.used == route_width(topk)).all()
    torch.cuda.synchronize()
