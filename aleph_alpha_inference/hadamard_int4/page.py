# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""The int4 page layout and the tables that place every dim's nibble in it.

One (block, head) page of ``BS`` tokens is ``BS * STATE_BYTES`` bytes in four regions:

    [ K codes: BS x 64 B | K terms: BS x 4 B | V codes: BS x 64 B | V terms: BS x 4 B ]

K and V are affine per (token, head): ``x ~= s * n + m`` with ``n`` in 0..15 and one
fp16 (s, m) pair per token and side. The codes are stored in the order the decode
kernel's ``mma.m16n8k16`` fragments consume them (see ``decode.py``), so each thread's
16-byte load of a K row holds exactly the nibbles of its own B fragments, and the two
tokens of a V fragment pair sit 16 bits apart in the same 32-bit word.
"""

import numpy as np
import torch

from aleph_alpha_inference.hadamard_int4 import STATE_BYTES

DIMS = 128


def _key_maps() -> tuple[np.ndarray, np.ndarray]:
    """K row byte b -> the dim of its low nibble, the dim of its high nibble.

    The decode kernel unpacks K rows with the B-fragment layout of ``mma.m16n8k16``
    (S = Q K^T takes B = K^T). Thread (g, c) owns a row's 16 bytes at ``16*c``, and its
    register p = 2*i + h (k-step i, register half h) is the fp16x2 pair of dims
    ``16*i + 8*h + 2*c + {0, 1}`` of column g = the token. Unpacking turns nibble
    ``p % 4`` into the register's low half and nibble ``p % 4 + 4`` into its high half,
    so a dim lands in word ``p // 4`` of the chunk at nibble ``(p % 4) + 4*e``.
    """
    lo = np.full(64, -1, dtype=np.int32)
    hi = np.full(64, -1, dtype=np.int32)
    for step in range(8):  # k-steps of the QK^T product
        for half in range(2):  # the step's two B registers
            word = (2 * step + half) // 4
            slot = (2 * step + half) % 4
            for col in range(4):  # lane column c: the chunk's bytes 16*c .. +16
                for e in range(2):  # the register pair's two consecutive dims
                    dim = 16 * step + 8 * half + 2 * col + e
                    nibble = slot + 4 * e
                    byte = 16 * col + 4 * word + nibble // 2
                    (hi if nibble % 2 else lo)[byte] = dim
    assert (lo >= 0).all() and (hi >= 0).all() and set(lo) | set(hi) == set(range(DIMS))
    return lo, hi


def _value_maps() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Token slot ``ts`` of a 16-token V block: its 64 bytes ->
    (block byte, dim lo, dim hi).

    The decode kernel's PV product takes B = V, 16 tokens x 8 dims per fragment. Thread
    (g, c) reads a block's 32 bytes at ``(4*g + c) * 32``: register p < 4 pairs tokens
    ``2*c + {0, 1}``, register p >= 4 pairs ``8 + 2*c + {0, 1}``, both over dims
    ``8*(4*(p % 4) + s) + g`` in nibble slot ``s`` (low token) and
    ``s + 4`` (high token).
    """
    pos = np.full((16, 64), -1, dtype=np.int32)
    lo = np.full((16, 64), -1, dtype=np.int32)
    hi = np.full((16, 64), -1, dtype=np.int32)
    owner: dict[int, int] = {}
    for ts in range(16):
        col = (ts % 8) // 2  # lane column that owns this token's bytes
        entries: dict[int, list] = {}
        for step in range(16):  # n-blocks of the PV product: dims 8*step .. +8
            for lane_row in range(8):  # fragment row g: dim 8*step + lane_row
                word = 4 * (ts // 8) + step // 4
                # even token takes low nibble slots 0-3; odd token takes high slots 4-7
                nibble = (step % 4) + 4 * (ts % 2)
                byte = (lane_row * 4 + col) * 32 + word * 4 + nibble // 2
                entries.setdefault(byte, [None, None])[nibble % 2] = 8 * step + lane_row
        assert len(entries) == 64
        for k, (byte, (dl, dh)) in enumerate(sorted(entries.items())):
            assert byte not in owner, "byte shared by two tokens"
            owner[byte] = ts
            pos[ts, k], lo[ts, k], hi[ts, k] = byte, dl, dh
    assert (lo >= 0).all() and (hi >= 0).all()
    assert set(lo.flatten()) | set(hi.flatten()) == set(range(DIMS))
    assert len(owner) == 1024
    return pos, lo, hi


_TABLES: dict[torch.device, dict[str, torch.Tensor]] = {}


def tables(device: torch.device) -> dict[str, torch.Tensor]:
    """The placement tables on ``device``, built once per device."""
    if device not in _TABLES:
        klo, khi = _key_maps()
        vpos, vlo, vhi = _value_maps()

        _TABLES[device] = {
            name: torch.tensor(map_, dtype=torch.int32, device=device)
            for name, map_ in (
                ("k_lo", klo),
                ("k_hi", khi),
                ("v_pos", vpos),
                ("v_lo", vlo),
                ("v_hi", vhi),
            )
        }
    return _TABLES[device]


def cache_views(kv_cache: torch.Tensor) -> dict[str, torch.Tensor]:
    """vLLM's per-layer int4 view ``[blocks, heads, BS, STATE_BYTES]`` (uint8, HND or
    NHD strides) -> the four regions of each (block, head) page.

    In either physical layout a block's used bytes (BS slots x H heads x STATE_BYTES)
    are one contiguous span at the block start. Every access to this cache is ours, so
    we split that span into per-head pages of BS * STATE_BYTES bytes ourselves."""
    blocks, heads, tokens, content = kv_cache.shape
    strides = kv_cache.stride()
    assert content == STATE_BYTES and kv_cache.dtype == torch.uint8 and strides[3] == 1
    span = (tokens - 1) * strides[2] + (heads - 1) * strides[1] + content
    assert span == tokens * heads * content and strides[0] >= span, (
        f"unexpected int4 KV layout {tuple(strides)}"
    )
    region = torch.as_strided(
        kv_cache, (blocks, heads, tokens * content), (strides[0], tokens * content, 1)
    )

    def terms(offset: int):
        return (
            region[..., offset : offset + tokens * 4]
            .view(torch.float16)
            .view(blocks, heads, tokens, 2)
        )

    return {
        "k_codes": region[..., : tokens * 64],
        "k_terms": terms(tokens * 64),
        "v_codes": region[..., tokens * 68 : tokens * 132],
        "v_terms": terms(tokens * 132),
    }


def alloc_cache(
    num_blocks: int, num_kv_heads: int, block_size: int, device
) -> dict[str, torch.Tensor]:
    """A standalone cache (tests) in the page layout."""
    raw = torch.zeros(
        num_blocks,
        num_kv_heads,
        block_size,
        STATE_BYTES,
        dtype=torch.uint8,
        device=device,
    )
    return cache_views(raw)
