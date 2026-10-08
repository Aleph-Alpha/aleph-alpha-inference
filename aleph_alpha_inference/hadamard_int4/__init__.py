# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""Hadamard-rotated int4 KV cache for Kolibri 1's full-attention layers.

Kolibri 1 alternates 40 sliding-window layers (RoPE, window 513) with 10
full-attention layers (no positional encoding). At long context the full layers
hold nearly all of the cache, so only they are compressed: their K and V are
stored as 4-bit affine codes with one fp16 scale/offset pair per token and head,
while the sliding-window layers keep the server's ``--kv-cache-dtype``.

Keys are quantized in a rotated basis, as in QuaRot (Ashkboos et al., 2024,
arXiv:2404.00456): a per-layer randomized Hadamard matrix ``R`` spreads the few
large channels of a key over all 128 dims before rounding, so no single channel
blows up the per-token scale. Kolibri's QK-norm is
``q = g_q * n(W_q x)``, ``k = g_k * n(W_k x)`` with ``n`` a scaleless RMS
normalize, and attention scores only see the product of the gains, so both are
moved to the query side before rotating:

    q = R diag(g_q g_k) n(W_q x)        k = R n(W_k x)

Scores are unchanged in exact arithmetic, the cached keys are gain-free
unit-RMS vectors, and ``diag(g_q g_k) R^T`` is one fixed matrix per layer, so
normalization, gains and rotation run as a single kernel over the QKV projection
output at about the cost of the stock QK-norm.

Enable with ``--additional-config '{"kolibri1": {"kv_cache": "hadamard-int4"}}'``;
an optional ``"hadamard_seed"`` picks the rotations. The option is part of
vLLM's config hash, so compiled graphs are never shared with the stock path.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from vllm.config import VllmConfig

HEAD_DIM = 128
# vLLM's KV dtype whose per-(token, head) state is 64 B of codes plus a 4 B fp16
# scale/offset pair per side -- exactly what this cache stores.
KV_DTYPE = "int4_per_token_head"
# Bytes of one (token, head) state: K codes 64 + K terms 4 + V codes 64 + V terms 4.
STATE_BYTES = 2 * (HEAD_DIM // 2 + 4)
# The decode kernel walks 32 tokens per iteration, so block sizes are multiples
# of it; `full_attention_block_size` never returns anything smaller.
DECODE_TILE = 32
# Smallest block size the sliding-window layers' attention kernels accept.
MIN_SLIDING_BLOCK = 16
DEFAULT_SEED = 1234


@dataclass(frozen=True)
class HadamardInt4Config:
    seed: int = DEFAULT_SEED


def get_config(vllm_config: "VllmConfig") -> HadamardInt4Config | None:
    """Read ``additional_config["kolibri1"]``; ``None`` unless the int4 KV cache
    is on."""
    options: Any = vllm_config.additional_config
    if not isinstance(options, dict):
        return None
    opts = options.get("kolibri1") or {}
    if not isinstance(opts, dict):
        raise TypeError(f"kolibri1 options must be an object, got {opts!r}")
    unknown = set(opts) - {"kv_cache", "hadamard_seed"}
    if unknown:
        raise ValueError(f"unknown kolibri1 options: {sorted(unknown)}")
    kind = opts.get("kv_cache")
    if kind is None:
        return None
    if kind != "hadamard-int4":
        raise ValueError(f"kolibri1.kv_cache must be 'hadamard-int4', got {kind!r}")
    return HadamardInt4Config(seed=int(opts.get("hadamard_seed", DEFAULT_SEED)))


def full_attention_block_size(
    sliding_bytes_per_token_head: int, min_block: int = 0
) -> int:
    """Smallest block size (multiple of DECODE_TILE, and not below min_block rounded up
    to one) whose int4 page covers one minimal sliding-window page.

    Every KV cache group shares one block pool with a single page size, and an int4 page
    (STATE_BYTES per token-head) is neither a divisor nor a multiple of an fp8
    (256 B) or bf16 (512 B) page, so the sliding pages are padded up to the int4 page:
    32 int4 tokens cover 16 fp8 tokens with 6% padding, 64 cover 16 bf16 tokens.
    """
    needed = MIN_SLIDING_BLOCK * sliding_bytes_per_token_head
    tokens = DECODE_TILE * -(-needed // (DECODE_TILE * STATE_BYTES))
    if min_block > tokens:
        tokens = DECODE_TILE * -(-min_block // DECODE_TILE)
    return tokens


def _sylvester(dim: int) -> torch.Tensor:
    """Hadamard matrix of order ``dim`` (a power of two), built by repeated doubling.

    Built on the CPU so every rank of a distributed run gets the same matrix."""
    h = torch.ones(1, 1, device="cpu")
    while h.shape[0] < dim:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h


def layer_rotation(layer_index: int, seed: int = DEFAULT_SEED) -> torch.Tensor:
    """``R = H diag(s) / sqrt(HEAD_DIM)``, an orthogonal fp32 [128, 128] matrix.

    The signs ``s`` are drawn per layer from a CPU generator seeded
    ``seed + layer_index`` and shared by all heads of the layer (GQA needs q and
    k in one basis). Built on the CPU, so every rank of a distributed run draws
    the same matrix whatever its default device.
    """
    rng = torch.Generator(device="cpu").manual_seed(seed + layer_index)
    signs = (
        torch.randint(0, 2, (HEAD_DIM,), generator=rng, device="cpu").float() * 2 - 1
    )
    return _sylvester(HEAD_DIM) * signs[None, :] / HEAD_DIM**0.5


def fold_norm_gains(
    q_gain: torch.Tensor, k_gain: torch.Tensor, rotation: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-vector maps applied after the scaleless RMS normalize: ``q_row @ M_q``,
    ``k_row @ M_k``, with ``M_q = diag(g_q g_k) R^T`` and ``M_k = R^T``, both fp16
    [128, 128].

    ``(q M_q) . (k M_k) = (g_q g_k q) (R R^T) k = (g_q q) . (g_k k)``, the stock score.
    """
    rotated = rotation.to(device=q_gain.device, dtype=torch.float32).T
    q_map = (q_gain.float() * k_gain.float())[:, None] * rotated
    return q_map.half().contiguous(), rotated.half().contiguous()


__all__ = [
    "DECODE_TILE",
    "DEFAULT_SEED",
    "HEAD_DIM",
    "KV_DTYPE",
    "MIN_SLIDING_BLOCK",
    "STATE_BYTES",
    "HadamardInt4Config",
    "fold_norm_gains",
    "full_attention_block_size",
    "get_config",
    "layer_rotation",
]
