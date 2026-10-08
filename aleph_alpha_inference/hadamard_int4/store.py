# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""Triton kernels for the int4 pages: pack (cache write), restore (the dequantized
history the prefill path attends over), and the split combine after the decode kernel.

Packing is affine per (token, head): ``x ~= s * n + m`` with ``n`` in 0..15, ``s`` the
fp16 range/15 and ``m`` the fp16 minimum. Both terms are rounded through fp16 so the
restore math quantizes with exactly the values that were stored. The codes land where
``page.py``'s tables put them, so the decode kernel's fragment loads need no repacking.
"""

import torch
from vllm.triton_utils import tl, triton

from aleph_alpha_inference.hadamard_int4 import page

# Range below this would quantize everything to one code and blow the relative error
# up on restore; all-zero rows (a whole constant row) get this scale instead.
MIN_SCALE = 1e-8

DIMS = 128


@triton.jit
def _terms(x, min_scale):
    """fp16 (scale, offset) of one row: the range over 15, floored at `min_scale`;
    both rounded through fp16, since the codes must quantize with the stored values."""
    low = tl.min(x, 0)
    high = tl.max(x, 0)
    scale = tl.maximum((high - low) / 15.0, min_scale).to(tl.float16).to(tl.float32)
    offset = low.to(tl.float16).to(tl.float32)
    return scale, offset


@triton.jit
def _pack_cell(
    x_ptr,
    codes_ptr,
    terms_ptr,
    lo_ptr,
    hi_ptr,
    pos_ptr,
    token,
    head,
    page,
    off,
    stride_xt,
    stride_xh,
    spb,
    sph,
    stb,
    sth,
    stt,
    min_scale,
    IS_VALUE: tl.constexpr,
):
    """Quantize one (token, head) row into its 64 code bytes and fp16 (scale,
    offset)."""
    src = x_ptr + token * stride_xt + head * stride_xh
    x = tl.load(src + tl.arange(0, 128)).to(tl.float32)
    scale, offset = _terms(x, min_scale)
    byte = tl.arange(0, 64)
    if IS_VALUE:
        # a token's bytes are scattered over its 16-token block, in fragment order
        ts = off % 16
        dlo = tl.load(lo_ptr + ts * 64 + byte)
        dhi = tl.load(hi_ptr + ts * 64 + byte)
        dst = (
            page * spb
            + head * sph
            + (off // 16) * 1024
            + tl.load(pos_ptr + ts * 64 + byte)
        )
    else:
        dlo = tl.load(lo_ptr + byte)
        dhi = tl.load(hi_ptr + byte)
        dst = page * spb + head * sph + off * 64 + byte
    xl = tl.load(src + dlo).to(tl.float32)
    xh = tl.load(src + dhi).to(tl.float32)
    nl = tl.minimum(tl.maximum(tl.floor((xl - offset) / scale + 0.5), 0.0), 15.0)
    nh = tl.minimum(tl.maximum(tl.floor((xh - offset) / scale + 0.5), 0.0), 15.0)
    tl.store(codes_ptr + dst, nl.to(tl.uint8) | (nh.to(tl.uint8) << 4))
    dst = page * stb + head * sth + off * stt
    tl.store(terms_ptr + dst, scale.to(tl.float16))
    tl.store(terms_ptr + dst + 1, offset.to(tl.float16))


@triton.jit
def _pack_kernel(
    k_ptr,
    v_ptr,
    slot_ptr,
    kc_ptr,
    kt_ptr,
    vc_ptr,
    vt_ptr,
    klo_ptr,
    khi_ptr,
    vlo_ptr,
    vhi_ptr,
    vpos_ptr,
    stride_kt,
    stride_kh,
    stride_vt,
    stride_vh,
    spb,
    sph,
    stb,
    sth,
    stt,
    min_scale,
    BS: tl.constexpr,
):
    """Grid (T, H, 2): axis 2 picks K (0) or V (1). Negative slots are padding."""
    token = tl.program_id(0)
    head = tl.program_id(1)
    slot = tl.load(slot_ptr + token).to(tl.int64)
    if slot < 0:
        return
    page_ = slot // BS
    off = slot % BS
    if tl.program_id(2) == 0:
        _pack_cell(
            k_ptr,
            kc_ptr,
            kt_ptr,
            klo_ptr,
            khi_ptr,
            vpos_ptr,
            token,
            head,
            page_,
            off,
            stride_kt,
            stride_kh,
            spb,
            sph,
            stb,
            sth,
            stt,
            min_scale,
            IS_VALUE=False,
        )
    else:
        _pack_cell(
            v_ptr,
            vc_ptr,
            vt_ptr,
            vlo_ptr,
            vhi_ptr,
            vpos_ptr,
            token,
            head,
            page_,
            off,
            stride_vt,
            stride_vh,
            spb,
            sph,
            stb,
            sth,
            stt,
            min_scale,
            IS_VALUE=True,
        )


def pack(
    key: torch.Tensor, value: torch.Tensor, slots: torch.Tensor, cache: dict
) -> None:
    """Pack key/value ``[T, H, 128]`` (last dim contiguous) into the cache at the flat
    token ids ``slots``; negative slots are padding and skipped."""
    tokens, heads, _ = key.shape
    if tokens == 0:
        return
    layout = page.tables(key.device)
    k_codes, k_terms = cache["k_codes"], cache["k_terms"]
    v_codes, v_terms = cache["v_codes"], cache["v_terms"]
    assert k_codes.stride() == v_codes.stride() and k_terms.stride() == v_terms.stride()
    assert key.stride(2) == 1 and value.stride(2) == 1
    _pack_kernel[(tokens, heads, 2)](
        key,
        value,
        slots,
        k_codes,
        k_terms,
        v_codes,
        v_terms,
        layout["k_lo"],
        layout["k_hi"],
        layout["v_lo"],
        layout["v_hi"],
        layout["v_pos"],
        key.stride(0),
        key.stride(1),
        value.stride(0),
        value.stride(1),
        k_codes.stride(0),
        k_codes.stride(1),
        k_terms.stride(0),
        k_terms.stride(1),
        k_terms.stride(2),
        MIN_SCALE,
        BS=k_terms.shape[2],
        num_warps=1,
    )


# Rows zeroed after the packed history: one FlashAttention K/V tile reads at most this
# many rows (FA2 and FA3 both stay within 256); its reads past the end are masked to
# probability 0, but 0 * NaN is NaN, so the rows it may touch must hold finite values.
RESTORE_PAD = 256


@triton.jit
def _restore_kernel(
    kc_ptr,
    kt_ptr,
    vc_ptr,
    vt_ptr,
    klo_ptr,
    khi_ptr,
    vlo_ptr,
    vhi_ptr,
    vpos_ptr,
    pages_ptr,
    cu_ptr,
    k_out,
    v_out,
    stride_pages,
    stride_out_t,
    stride_out_h,
    max_len,
    num_seqs,
    total_rows,
    spb,
    sph,
    stb,
    sth,
    stt,
    BS: tl.constexpr,
):
    """Grid (num_seqs * cdiv(max_len, 16) + RESTORE_PAD // 16, H). Sequence i's token j
    restores to row cu[i] + j; the last programs zero the pad rows after the packed
    history."""
    program = tl.program_id(0)
    head = tl.program_id(1)
    tiles = tl.cdiv(max_len, 16)
    if program >= num_seqs * tiles:
        rows = (
            tl.load(cu_ptr + num_seqs)
            + (program - num_seqs * tiles) * 16
            + tl.arange(0, 16)
        )
        live = rows < total_rows
        dim = tl.arange(0, 128)
        zero = tl.zeros([16, 128], k_out.dtype.element_ty)
        dst = (
            rows.to(tl.int64)[:, None] * stride_out_t
            + head * stride_out_h
            + dim[None, :]
        )
        tl.store(k_out + dst, zero, mask=live[:, None])
        tl.store(v_out + dst, zero, mask=live[:, None])
        return
    seq = program // tiles
    first = (program % tiles) * 16
    row0 = tl.load(cu_ptr + seq)
    length = tl.load(cu_ptr + seq + 1) - row0
    if first >= length:
        return
    ts = tl.arange(0, 16)
    token = first + ts
    live = token < length
    page_ = tl.load(pages_ptr + seq * stride_pages + first // BS).to(tl.int64)
    off0 = first % BS
    byte = tl.arange(0, 64)
    row = (row0 + token).to(tl.int64)
    # K: one 64-byte code row per token, terms next to the codes in the page
    src = page_ * spb + head * sph + (off0 + ts)[:, None] * 64 + byte[None, :]
    packed = tl.load(kc_ptr + src, mask=live[:, None], other=0)
    terms = kt_ptr + page_ * stb + head * sth + (off0 + ts) * stt
    scale = tl.load(terms, mask=live, other=0.0).to(tl.float32)
    offset = tl.load(terms + 1, mask=live, other=0.0).to(tl.float32)
    dst = k_out + row[:, None] * stride_out_t + head * stride_out_h
    dlo = tl.load(klo_ptr + byte)
    dhi = tl.load(khi_ptr + byte)
    tl.store(
        dst + dlo[None, :],
        ((packed & 0xF).to(tl.float32) * scale[:, None] + offset[:, None]).to(
            k_out.dtype.element_ty
        ),
        mask=live[:, None],
    )
    tl.store(
        dst + dhi[None, :],
        ((packed >> 4).to(tl.float32) * scale[:, None] + offset[:, None]).to(
            k_out.dtype.element_ty
        ),
        mask=live[:, None],
    )
    # V: one 1024-byte block per 16 tokens; pages hold a multiple of 32 tokens, so a
    # 16-token tile never straddles a page
    src = (
        page_ * spb
        + head * sph
        + (off0 // 16) * 1024
        + tl.load(vpos_ptr + ts[:, None] * 64 + byte[None, :])
    )
    packed = tl.load(vc_ptr + src, mask=live[:, None], other=0)
    terms = vt_ptr + page_ * stb + head * sth + (off0 + ts) * stt
    scale = tl.load(terms, mask=live, other=0.0).to(tl.float32)
    offset = tl.load(terms + 1, mask=live, other=0.0).to(tl.float32)
    dst = v_out + row[:, None] * stride_out_t + head * stride_out_h
    vlo = tl.load(vlo_ptr + ts[:, None] * 64 + byte[None, :])
    vhi = tl.load(vhi_ptr + ts[:, None] * 64 + byte[None, :])
    tl.store(
        dst + vlo,
        ((packed & 0xF).to(tl.float32) * scale[:, None] + offset[:, None]).to(
            v_out.dtype.element_ty
        ),
        mask=live[:, None],
    )
    tl.store(
        dst + vhi,
        ((packed >> 4).to(tl.float32) * scale[:, None] + offset[:, None]).to(
            v_out.dtype.element_ty
        ),
        mask=live[:, None],
    )


def restore(
    cache: dict,
    block_table: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_len: int,
    dtype=torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Restore sequences of up to ``max_len`` tokens, packed back to back at rows
    ``cu_seqlens[i] ...`` (device tensor, ``[n + 1]``), into K, V of shape
    ``[n * max_len + RESTORE_PAD, H, 128]``.

    The pad rows after ``cu_seqlens[-1]`` are zeroed for attention kernels that read
    whole tiles; rows beyond the buffer are left unwritten. Every size comes from the
    host-side bound ``max_len``: no device sync.
    """
    num_seqs = block_table.shape[0]
    heads = cache["k_codes"].shape[1]
    tokens = cache["k_terms"].shape[2]
    k = torch.empty(
        num_seqs * max_len + RESTORE_PAD,
        heads,
        DIMS,
        device=block_table.device,
        dtype=dtype,
    )
    v = torch.empty_like(k)
    if num_seqs == 0:
        return k, v
    layout = page.tables(block_table.device)
    _restore_kernel[(num_seqs * triton.cdiv(max_len, 16) + RESTORE_PAD // 16, heads)](
        cache["k_codes"],
        cache["k_terms"],
        cache["v_codes"],
        cache["v_terms"],
        layout["k_lo"],
        layout["k_hi"],
        layout["v_lo"],
        layout["v_hi"],
        layout["v_pos"],
        block_table,
        cu_seqlens,
        k,
        v,
        block_table.stride(0),
        k.stride(0),
        k.stride(1),
        max_len,
        num_seqs,
        k.shape[0],
        cache["k_codes"].stride(0),
        cache["k_codes"].stride(1),
        cache["k_terms"].stride(0),
        cache["k_terms"].stride(1),
        cache["k_terms"].stride(2),
        BS=tokens,
        num_warps=4,
    )
    return k, v


@triton.jit
def _combine_kernel(
    partials_ptr,
    lses_ptr,
    out_ptr,
    stride_out_b,
    stride_out_h,
    KV_HEADS: tl.constexpr,
    GROUP: tl.constexpr,
    SPLITS: tl.constexpr,
    CHUNK: tl.constexpr,
):
    """out[b, hq] = sum_s softmax(lse)_s * partials[b, hq // GROUP, hq % GROUP, s].

    The lses are in the log2 domain (the decode kernel's exp2 softmax), so the weights
    are exp2(lse - max)."""
    batch = tl.program_id(0)
    head_q = tl.program_id(1)
    row = (batch * KV_HEADS + head_q // GROUP) * 16 + head_q % GROUP
    slot = tl.arange(0, CHUNK)
    dim = tl.arange(0, 128)
    peak = tl.full([CHUNK], float("-inf"), tl.float32)
    for start in range(0, SPLITS, CHUNK):
        peak = tl.maximum(
            peak,
            tl.load(
                lses_ptr + row * SPLITS + start + slot,
                mask=start + slot < SPLITS,
                other=float("-inf"),
            ),
        )
    top = tl.max(peak, 0)
    top = tl.where(top == float("-inf"), 0.0, top)
    weights = tl.zeros([CHUNK], tl.float32)
    acc = tl.zeros([128], tl.float32)
    for start in range(0, SPLITS, CHUNK):
        live = start + slot < SPLITS
        lse = tl.load(
            lses_ptr + row * SPLITS + start + slot, mask=live, other=float("-inf")
        )
        weight = tl.where(lse > float("-inf"), tl.exp2(lse - top), 0.0)
        weights += weight
        partial = tl.load(
            partials_ptr + (row * SPLITS + start + slot)[:, None] * 128 + dim[None, :],
            mask=live[:, None],
            other=0.0,
        )
        acc += tl.sum(partial * weight[:, None], 0)
    out = acc / tl.maximum(tl.sum(weights, 0), 1e-30)
    tl.store(
        out_ptr + batch * stride_out_b + head_q * stride_out_h + dim,
        out.to(out_ptr.dtype.element_ty),
    )


def combine(
    partials: torch.Tensor,
    lses: torch.Tensor,
    out: torch.Tensor,
    num_kv_heads: int,
    group: int,
) -> None:
    """Merge the decode kernel's per-split partial outputs and lses into ``out``."""
    batches, heads_q = out.shape[0], out.shape[1]
    assert out.stride(-1) == 1
    _combine_kernel[(batches, heads_q)](
        partials,
        lses,
        out,
        out.stride(0),
        out.stride(1),
        KV_HEADS=num_kv_heads,
        GROUP=group,
        SPLITS=partials.shape[3],
        CHUNK=32,
    )


def pack_reference(x: torch.Tensor) -> torch.Tensor:
    """What the pack kernel stores for a row, restored to fp32."""
    x = x.float()
    low, high = x.amin(-1, keepdim=True), x.amax(-1, keepdim=True)
    scale = ((high - low) / 15).clamp_min(MIN_SCALE).half().float()
    offset = low.half().float()
    return torch.floor((x - offset) / scale + 0.5).clamp(0, 15) * scale + offset


def restore_reference(
    cache: dict, block_table_row: torch.Tensor, length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-torch restore of the first ``length`` tokens of one sequence: fp32 K, V
    ``[length, H, 128]``."""
    layout = page.tables(cache["k_codes"].device)
    tokens = cache["k_terms"].shape[2]
    heads = cache["k_codes"].shape[1]
    token = torch.arange(length, device=cache["k_codes"].device)
    slot = token % tokens
    page_ = block_table_row[token // tokens].long()
    k_bytes = cache["k_codes"].unflatten(-1, (tokens, 64))[page_, :, slot]
    k = torch.empty(length, heads, DIMS, device=k_bytes.device)
    k[:, :, layout["k_lo"].long()] = (k_bytes & 0xF).float()
    k[:, :, layout["k_hi"].long()] = (k_bytes >> 4).float()
    block = cache["v_codes"].unflatten(-1, (tokens // 16, 1024))[page_, :, slot // 16]
    ts = (slot % 16).long()
    gathered = torch.gather(
        block, 2, layout["v_pos"].long()[ts][:, None, :].expand(-1, heads, -1)
    )
    v = torch.empty(length, heads, DIMS, device=k_bytes.device)
    v.scatter_(
        2,
        layout["v_lo"].long()[ts][:, None, :].expand(-1, heads, -1),
        (gathered & 0xF).float(),
    )
    v.scatter_(
        2,
        layout["v_hi"].long()[ts][:, None, :].expand(-1, heads, -1),
        (gathered >> 4).float(),
    )
    k_terms = cache["k_terms"][page_, :, slot].float()
    v_terms = cache["v_terms"][page_, :, slot].float()
    return k * k_terms[..., :1] + k_terms[..., 1:], v * v_terms[..., :1] + v_terms[
        ..., 1:
    ]
