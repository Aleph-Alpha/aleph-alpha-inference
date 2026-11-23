# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""Sparse prefill attention over paged NVFP4 K/V, without staging the context.

See `nvfp4_attention.cu`: one CTA per query row gathers each selected token's
K/V for all four KV groups once and dequantizes it into FP16 MMA fragments
(FP32 accumulation). Outputs match BF16 attention over the dequantized cache
to within its own rounding error.
"""

from functools import lru_cache
from pathlib import Path

import torch

from aleph_alpha_inference.kernels.arch import gencode
from aleph_alpha_inference.kernels.native_build import load_extension


@lru_cache(None)
def native():
    """The compiled extension, built on first use."""
    return load_extension(
        "nvfp4_attention",
        [Path(__file__).with_suffix(".cu")],
        extra_cuda_cflags=[
            "-O3",
            "-lineinfo",
            gencode(),
        ],
    )


def nvfp4_attention(
    q,
    cache,
    table,
    requests,
    route,
    used,
    *,
    out=None,
    only_tiles=None,
    tile_rows=0,
):
    """Attention of each query row over its `used` selected keys.

    `q` and `out` are [rows, 1, 48, 128] or [rows, 48, 128] BF16; `route`
    holds logical key positions of request `requests[row]`, whose pages are
    `table[request]` in `cache`, an NVFP4 cache of 128-token pages whose
    page stride is a multiple of 256 bytes. Rows without keys get zeros.
    The data of `q`, `out` and `cache.storage` must start at 16-byte aligned
    addresses; this is not checked.
    With `only_tiles` ([ceil(rows / tile_rows)] int32), only rows of nonzero
    tiles are computed; the others keep `out` as it is.
    """
    if out is None:
        out = torch.empty_like(q)
    native().nvfp4_attention(
        _rows(q),
        cache.storage,
        table,
        requests,
        route,
        used,
        _rows(out),
        only_tiles,
        tile_rows,
        128**-0.5,
    )
    return out


def _rows(x):
    return x[:, 0] if x.ndim == 4 else x
