# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""Sparse prefill as FA4 over per-tile key unions of a staged BF16 context.

Consecutive queries of one request select overlapping keys. `union_tiles`
gives each tile of TILE_ROWS queries the sorted union of their keys and a
per-query bitmask over it; vLLM's FA4 then gathers the union rows of the
request's BF16-dequantized context (paged KV with one-token pages) and masks
each query to its own keys through a mask_mod over the tile's bitmask.
"""

from functools import lru_cache
from pathlib import Path

import cutlass
import cutlass.cute as cute
import torch

from aleph_alpha_inference.kernels.arch import gencode, is_blackwell
from aleph_alpha_inference.kernels.native_build import load_extension

# 21 queries x 12 heads = 252 packed rows fill FA4's two 128-row Q stages.
TILE_ROWS = 21
# Mask rows per tile: one bit of a word per row.
MASK_ROWS = 32
# On B200, FA4 over unions costs ~0.8 ms per 1,000 mean union keys per 32k
# queries, plus ~0.6 ms to build tiles and stage; the
# per-row NVFP4 kernel ~9.5 ms (2,184 keys). On H100 the element-wise mask
# makes it ~2.4 ms per 1,000 keys against ~23 ms per row kernel. Requests
# whose tiles average more than `union_gate()` keys run on the per-row
# kernel; so do single tiles above UNION_LIMIT, the workspace capacity.
UNION_LIMIT = 16384


@lru_cache(None)
def union_gate():
    """Mean union keys per tile above which a request uses the per-row kernel."""
    return 11000 if is_blackwell() else 9000


@lru_cache(None)
def native():
    """The `union_tiles.cu` extension, built on first use."""
    return load_extension(
        "union_tiles",
        [Path(__file__).parent / "union_tiles.cu"],
        extra_cuda_cflags=[
            "-O3",
            "-lineinfo",
            gencode(),
        ],
    )


@cute.jit
def bit_mask_mod(
    batch: cute.TensorSSA,
    head: cute.TensorSSA,
    q_idx: cute.TensorSSA,
    kv_idx: cute.TensorSSA,
    seqlen_info,
    aux_tensors: list,
) -> cute.TensorSSA:
    """One bit of the tile masks per (query, union key): FA4's SM90 kernel
    takes only element-wise mask_mods, Blackwell's reads 128 keys at once
    (vLLM's `dense_mask_mod`)."""
    from vllm.vllm_flash_attn.cute import utils

    mask = aux_tensors[0]
    tile = utils.ssa_to_scalar(batch)
    row = utils.ssa_to_scalar(q_idx)
    key = utils.ssa_to_scalar(kv_idx)
    word = cutlass.Uint32(mask[tile, row, key >> 5])
    bit = cute.make_rmem_tensor(1, dtype=cutlass.Uint32)
    bit[0] = utils.shr_u32(word, cutlass.Uint32(key & 31)) & 1
    return bit.load()


def _mask_mod():
    if is_blackwell():
        from vllm.model_executor.layers.attention.sparse_mla_mask import dense_mask_mod

        return dense_mask_mod
    return bit_mask_mod


def union_workspace(rows, device):
    """Scratch of `union_tiles` for up to `rows` queries.

    Per tile: union keys, union length, fallback flag, rows for FA4 and the
    row masks. The last tensor holds the sum of the union lengths.
    """
    tiles = (rows + TILE_ROWS - 1) // TILE_ROWS
    words = UNION_LIMIT // 32
    return (
        torch.empty(tiles, UNION_LIMIT, device=device, dtype=torch.int32),
        torch.empty(tiles, device=device, dtype=torch.int32),
        torch.empty(tiles, device=device, dtype=torch.int32),
        torch.empty(tiles, device=device, dtype=torch.int32),
        torch.empty(tiles, MASK_ROWS, words, device=device, dtype=torch.int32),
        torch.empty(1, device=device, dtype=torch.int32),
    )


def union_tiles(route, used, span, workspace):
    """Build the key union and the row masks of every tile of TILE_ROWS rows.

    Tiles with more than `UNION_LIMIT` union keys are left to the per-row
    kernel, and so are all tiles if they average more than `union_gate()` keys.

    Args:
        route: Logical key positions of one request's rows, [rows, width]
            int32, all below `span`.
        used: Number of valid entries of each `route` row, int32.
        span: Upper bound of the key positions.
        workspace: A `union_workspace` of at least `rows` rows.

    Returns:
        Per tile: the sorted union keys, the union length, the fallback flag,
        the number of rows FA4 attends (0 for tiles left to the per-row
        kernel) and the row masks over the union.
    """
    keys, count, fallback, qrows, mask, union_sum = workspace
    tiles = (route.shape[0] + TILE_ROWS - 1) // TILE_ROWS
    native().union_tiles(
        route,
        used,
        TILE_ROWS,
        span,
        UNION_LIMIT,
        union_gate(),
        keys,
        count,
        fallback,
        qrows,
        mask,
        union_sum,
    )
    return keys[:tiles], count[:tiles], fallback[:tiles], qrows[:tiles], mask[:tiles]


def union_attention(q, k, v, tiles, *, out=None):
    """FA4 over the union `tiles` (from `union_tiles`) of one request's rows.

    `q`: [rows, 48, 128]; `k`, `v`: the request's context [tokens, 1, 4, 128].
    Rows of spilled tiles are not written (FA4 attends `qrows` per tile).
    """
    from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd

    keys, count, _, qrows, mask = tiles
    rows = q.shape[0]
    if out is None:
        out = torch.zeros_like(q)
    cu = torch.arange(
        0, keys.shape[0] * TILE_ROWS + 1, TILE_ROWS, device=q.device, dtype=torch.int32
    )
    cu[-1] = rows
    _flash_attn_fwd(
        q,
        k,
        v,
        cu_seqlens_q=cu,
        seqused_q=qrows,
        seqused_k=count,
        max_seqlen_q=TILE_ROWS,
        max_seqlen_k=keys.shape[1],
        page_table=keys,
        causal=False,
        mask_mod=_mask_mod(),
        aux_tensors=[mask],
        aux_tensor_leading_dims=[2],
        # A fixed schedule keeps one executable for every row count.
        num_splits=1,
        out=out,
    )
    return out


@lru_cache(None)
def warm_up(device):
    """Compile FA4's two union executables (its varlen scheduler switches at
    256 tiles) before serving, on throwaway inputs."""
    from vllm.vllm_flash_attn.cute.interface import BIN_BATCH_SEARCH_THRESH

    native()
    for tiles in (1, BIN_BATCH_SEARCH_THRESH + 1):
        rows = tiles * TILE_ROWS
        span = 1024
        route = torch.arange(64, device=device, dtype=torch.int32)[None].repeat(rows, 1)
        used = torch.full((rows,), 64, device=device, dtype=torch.int32)
        q = torch.zeros(rows, 48, 128, device=device, dtype=torch.bfloat16)
        kv = torch.zeros(2, span, 1, 4, 128, device=device, dtype=torch.bfloat16)
        workspace = union_workspace(rows, device)
        union_attention(q, kv[0], kv[1], union_tiles(route, used, span, workspace))
    torch.cuda.synchronize(device)
