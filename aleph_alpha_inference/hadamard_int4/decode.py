# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""int4 KV decode attention in the CuTe DSL: nibbles stream from global memory straight
into ``mma.m16n8k16`` fragments.

The pages store codes in fragment order (`page.py`), so a thread's 16-byte load of a K
row is exactly its own B fragments, and one ``lop3`` plus ``sub.f16x2`` per register
turns the nibbles into exact fp16 values in place (Marlin's unpack). Nothing touches
shared memory, so the int4 cache streams at close to FlashAttention-3's fp8 bandwidth;
a Triton kernel, which has to stage unpacked tiles through shared memory, reaches about
half of it.

Thread roles follow mma.m16n8k16 (lane l: g = l // 4, c = l % 4):
  A (16 x 16, rows = the q heads of one kv head, padded to 16): a k-step's fragment
      covers dims 16*i + 2*c + {0,1} of rows g and g+8, then 16*i + 8 + 2*c + {0,1}.
  B (16 x 8, k x n), C (16 x 8): B registers hold k = 2*c + {0,1} and
      2*c + 8 + {0,1} of column n = g; C registers hold rows g, g+8 of columns
      2*c, 2*c+1.

The affine terms never touch the tensor cores: with x = s n + m,
  score = s (q . n) + m sum(q),        out = (p s) @ n + sum(p) m
so the MMAs run on the raw nibbles and (s, m) enter as per-token scalars.

Grid (B, kv heads, CTAs), 4 warps per CTA; every warp is an independent flash-decoding
split with its own online softmax over a token range, combined by `store.combine`.
"""

import math

import cutlass
import torch
from cutlass import Float32, Int32, const_expr, cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from vllm.triton_utils import triton

from aleph_alpha_inference.hadamard_int4 import store

DIMS = 128

_ATT = llvm.AsmDialect.AD_ATT


@dsl_user_op
def _load_quad(addr, *, loc=None, ip=None):
    """``ld.global.nc.v4.u32`` from an i64 address -> 4 x Int32."""
    quad = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [cutlass.Int64(addr).ir_value(loc=loc, ip=ip)],
        "ld.global.nc.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=_ATT,
    )
    return [
        Int32(llvm.extractvalue(T.i32(), quad, [k], loc=loc, ip=ip)) for k in range(4)
    ]


@dsl_user_op
def _load_word(addr, *, loc=None, ip=None):
    """``ld.global.nc.u32`` from an i64 address -> Int32."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [cutlass.Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.nc.u32 $0, [$1];",
            "=r,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=_ATT,
        )
    )


def _nibbles(word, slot):
    """fp16x2 of nibble `slot` of a packed word, as Int32.

    ``lop3`` copies the nibble and the one 16 bits above it into the mantissas of
    1024.0; ``sub.f16x2`` removes the bias. Exact 0..15 fp16 values, in place."""

    @dsl_user_op
    def unpack(w, *, loc=None, ip=None):
        return Int32(
            llvm.inline_asm(
                T.i32(),
                [Int32(w).ir_value(loc=loc, ip=ip)],
                "{\n.reg .b32 t, bias;\n"
                f"shr.u32 t, $1, {4 * slot};\n"
                "lop3.b32 t, t, 0x000f000f, 0x64006400, 0xea;\n"
                "mov.b32 bias, 0x64006400;\n"
                "sub.f16x2 $0, t, bias;\n}\n",
                "=r,r",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=_ATT,
            )
        )

    return unpack(word)


@dsl_user_op
def _mma(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, *, loc=None, ip=None):
    """mma.sync.m16n8k16 row.col, f32 += f16 * f16, all operands as registers."""
    quad = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [Int32(x).ir_value(loc=loc, ip=ip) for x in (a0, a1, a2, a3, b0, b1)]
        + [Float32(x).ir_value(loc=loc, ip=ip) for x in (c0, c1, c2, c3)],
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0, $1, $2, $3}, "
        "{$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=_ATT,
    )
    return [
        Float32(llvm.extractvalue(T.f32(), quad, [k], loc=loc, ip=ip)) for k in range(4)
    ]


@dsl_user_op
def _half2_to_floats(pair, *, loc=None, ip=None):
    """fp16x2 (as Int32) -> (Float32 low, Float32 high)."""
    pair_out = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Int32(pair).ir_value(loc=loc, ip=ip)],
        "{\n.reg .b16 lo, hi;\nmov.b32 {lo, hi}, $2;\n"
        "cvt.f32.f16 $0, lo;\ncvt.f32.f16 $1, hi;\n}\n",
        "=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=_ATT,
    )
    return Float32(llvm.extractvalue(T.f32(), pair_out, [0], loc=loc, ip=ip)), Float32(
        llvm.extractvalue(T.f32(), pair_out, [1], loc=loc, ip=ip)
    )


@dsl_user_op
def _floats_to_half2(low, high, *, loc=None, ip=None):
    """(Float32, Float32) -> fp16x2 as Int32 (low in the low half)."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(low).ir_value(loc=loc, ip=ip),
                Float32(high).ir_value(loc=loc, ip=ip),
            ],
            "cvt.rn.f16x2.f32 $0, $2, $1;",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=_ATT,
        )
    )


@dsl_user_op
def _bf16_pair_to_fp16(pair, *, loc=None, ip=None):
    """bf16x2 (as Int32) -> fp16x2: widen each half to f32 (shift / mask), then one
    ``cvt.rn.f16x2.f32``. Lets the kernel take the model's bf16 queries without a
    separate cast."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(pair).ir_value(loc=loc, ip=ip)],
            "{\n.reg .b32 lo, hi;\nshl.b32 lo, $1, 16;\nand.b32 hi, $1, 0xffff0000;\n"
            "cvt.rn.f16x2.f32 $0, hi, lo;\n}\n",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=_ATT,
        )
    )


def _element_ptr(tensor: cute.Tensor, coord):
    return (tensor.iterator + cute.crd2idx(coord, tensor.layout)).toint()


class DecodeKernel:
    """Grid (B, KV_HEADS, CTAS), 4 warps per CTA; warp w of CTA z is split z*4 + w of
    the SPLITS = CTAS*4 flash-decoding splits."""

    def __init__(self, group: int, page_tokens: int, query_bf16: bool = False):
        self.group = group  # q heads per kv head (<= 16)
        self.page_tokens = page_tokens  # page size, a multiple of 32
        self.query_bf16 = query_bf16  # m_q holds bf16 pairs: fp16 in the prologue

    @cute.jit
    def __call__(
        self,
        m_q: cute.Tensor,
        m_k: cute.Tensor,
        m_v: cute.Tensor,
        m_k_terms: cute.Tensor,
        m_v_terms: cute.Tensor,
        m_pages: cute.Tensor,
        m_lens: cute.Tensor,
        m_partials: cute.Tensor,
        m_lses: cute.Tensor,
        scale_log2: Float32,
        ctas: Int32,
        stream,
    ):
        batches = m_pages.shape[0]
        kv_heads = m_k.shape[1]
        self.kernel(
            m_q,
            m_k,
            m_v,
            m_k_terms,
            m_v_terms,
            m_pages,
            m_lens,
            m_partials,
            m_lses,
            scale_log2,
            ctas,
        ).launch(grid=[batches, kv_heads, ctas], block=[128, 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self,
        m_q: cute.Tensor,
        m_k: cute.Tensor,
        m_v: cute.Tensor,
        m_k_terms: cute.Tensor,
        m_v_terms: cute.Tensor,
        m_pages: cute.Tensor,
        m_lens: cute.Tensor,
        m_partials: cute.Tensor,
        m_lses: cute.Tensor,
        scale_log2: Float32,
        ctas: Int32,
    ):
        # m_q [B, HQ, 64] Int32 (fp16 pairs, or bf16 pairs if query_bf16)
        # m_k, m_v [P, KV_HEADS, BS*16] Int32 |
        # m_k_terms, m_v_terms [P, KV_HEADS, BS] Int32
        # m_pages [B, pages] | m_lens [B]
        # m_partials [B, KV_HEADS, 16, S, 128] f32 | m_lses [B, KV_HEADS, 16, S] f32
        group = self.group
        page_tokens = self.page_tokens
        thread, _, _ = cute.arch.thread_idx()
        batch, head, split_cta = cute.arch.block_idx()
        warp = thread // 32
        lane = thread % 32
        row = lane // 4
        col = lane % 4
        splits = ctas * 4
        split = split_cta * 4 + warp
        length = m_lens[batch]
        # the tile's token span, rounded to the 32-token walk so tiles never
        # straddle pages
        span = ((length + 31) // 32 + splits - 1) // splits * 32
        first = split * span
        last = cutlass.min(length, first + span)

        regs = lambda n: cute.make_layout((n,), stride=(1,))
        q_frag = cute.make_rmem_tensor(
            regs(32), Int32
        )  # Q A-fragments: 8 k-steps x 4 regs
        tally = cute.make_rmem_tensor(
            regs(8), Float32
        )  # qsum[2], max[2], denom[2], voffset[2]
        out_acc = cute.make_rmem_tensor(regs(64), Float32)  # O: 16 n-blocks x 4 regs
        score = cute.make_rmem_tensor(regs(16), Float32)  # S: 4 n-blocks x 4 regs
        p_frag = cute.make_rmem_tensor(
            regs(8), Int32
        )  # P A-fragments: 2 k-steps x 4 regs
        rescale = cute.make_rmem_tensor(regs(2), Float32)
        out_acc.fill(0.0)
        for half in cutlass.range_constexpr(2):
            tally[half] = Float32(0.0)
            tally[2 + half] = Float32(-math.inf)
            tally[4 + half] = Float32(0.0)
            tally[6 + half] = Float32(0.0)

        # ---- Q A-fragments: rows `row` and `row + 8` hold q heads
        # `head * group + row`, zero-padded to the fragment's 16 rows, plus the per-row
        # q sums shared by lanes
        for step in cutlass.range_constexpr(8):
            for half in cutlass.range_constexpr(2):
                q_row = row + 8 * half
                for pair in cutlass.range_constexpr(2):
                    q_pair = Int32(0)
                    if q_row < group:
                        q_pair = m_q[
                            batch, head * group + q_row, 8 * step + col + 4 * pair
                        ]
                        if const_expr(self.query_bf16):
                            q_pair = _bf16_pair_to_fp16(q_pair)
                    q_frag[4 * step + half + 2 * pair] = q_pair
                    low, high = _half2_to_floats(q_pair)
                    tally[half] = tally[half] + low + high
        for half in cutlass.range_constexpr(2):
            tally[half] = tally[half] + cute.arch.shuffle_sync_bfly(
                tally[half], offset=1
            )
            tally[half] = tally[half] + cute.arch.shuffle_sync_bfly(
                tally[half], offset=2
            )

        # K and V words plus terms of the next tile are prefetched one iteration ahead
        # (double buffered in registers), so global latency hides behind the current
        # tile's MMAs
        next_k = cute.make_rmem_tensor(
            regs(16), Int32
        )  # next tile: 4 n8 blocks x 4 words
        next_terms = cute.make_rmem_tensor(
            regs(8), Int32
        )  # next tile: K terms of C tokens
        next_v = cute.make_rmem_tensor(
            regs(16), Int32
        )  # next tile: 2 k-steps x 8 words
        next_v_terms = cute.make_rmem_tensor(regs(8), Int32)  # next tile: V terms
        cur_k = cute.make_rmem_tensor(regs(16), Int32)
        cur_terms = cute.make_rmem_tensor(regs(8), Int32)
        cur_v = cute.make_rmem_tensor(regs(16), Int32)
        cur_v_terms = cute.make_rmem_tensor(regs(8), Int32)

        if first < last:
            page_ = m_pages[batch, first // page_tokens]
            off = first % page_tokens
            for nb in cutlass.range_constexpr(4):
                quad = _load_quad(
                    _element_ptr(
                        m_k, (page_, head, (off + nb * 8 + row) * 16 + col * 4)
                    )
                )
                for w in cutlass.range_constexpr(4):
                    next_k[4 * nb + w] = quad[w]
                for e in cutlass.range_constexpr(2):
                    next_terms[2 * nb + e] = _load_word(
                        _element_ptr(
                            m_k_terms, (page_, head, off + nb * 8 + 2 * col + e)
                        )
                    )
                    next_v_terms[2 * nb + e] = _load_word(
                        _element_ptr(
                            m_v_terms, (page_, head, off + nb * 8 + 2 * col + e)
                        )
                    )
            for kk in cutlass.range_constexpr(2):
                v_row = (off + kk * 16) // 16 * 256 + (row * 4 + col) * 8
                v0 = _load_quad(_element_ptr(m_v, (page_, head, v_row)))
                v1 = _load_quad(_element_ptr(m_v, (page_, head, v_row + 4)))
                for w in cutlass.range_constexpr(4):
                    next_v[8 * kk + w] = v0[w]
                    next_v[8 * kk + 4 + w] = v1[w]

        for start in cutlass.range(first, last, 32):
            for k in cutlass.range_constexpr(16):
                cur_k[k] = next_k[k]
            for k in cutlass.range_constexpr(8):
                cur_terms[k] = next_terms[k]
                cur_v_terms[k] = next_v_terms[k]
            for k in cutlass.range_constexpr(16):
                cur_v[k] = next_v[k]
            # prefetch the next tile's K and V
            following = start + 32
            if following < last:
                page_ = m_pages[batch, following // page_tokens]
                off = following % page_tokens
                for nb in cutlass.range_constexpr(4):
                    quad = _load_quad(
                        _element_ptr(
                            m_k, (page_, head, (off + nb * 8 + row) * 16 + col * 4)
                        )
                    )
                    for w in cutlass.range_constexpr(4):
                        next_k[4 * nb + w] = quad[w]
                    for e in cutlass.range_constexpr(2):
                        next_terms[2 * nb + e] = _load_word(
                            _element_ptr(
                                m_k_terms, (page_, head, off + nb * 8 + 2 * col + e)
                            )
                        )
                        next_v_terms[2 * nb + e] = _load_word(
                            _element_ptr(
                                m_v_terms, (page_, head, off + nb * 8 + 2 * col + e)
                            )
                        )
                for kk in cutlass.range_constexpr(2):
                    v_row = (off + kk * 16) // 16 * 256 + (row * 4 + col) * 8
                    v0 = _load_quad(_element_ptr(m_v, (page_, head, v_row)))
                    v1 = _load_quad(_element_ptr(m_v, (page_, head, v_row + 4)))
                    for w in cutlass.range_constexpr(4):
                        next_v[8 * kk + w] = v0[w]
                        next_v[8 * kk + 4 + w] = v1[w]
            # ---- S = Q K^T over the tile's 32 tokens (4 n8 blocks)
            score.fill(0.0)
            for nb in cutlass.range_constexpr(4):
                for step in cutlass.range_constexpr(8):
                    b0 = _nibbles(cur_k[4 * nb + (2 * step) // 4], (2 * step) % 4)
                    b1 = _nibbles(
                        cur_k[4 * nb + (2 * step + 1) // 4], (2 * step + 1) % 4
                    )
                    acc = _mma(
                        q_frag[4 * step],
                        q_frag[4 * step + 1],
                        q_frag[4 * step + 2],
                        q_frag[4 * step + 3],
                        b0,
                        b1,
                        score[4 * nb],
                        score[4 * nb + 1],
                        score[4 * nb + 2],
                        score[4 * nb + 3],
                    )
                    for k in cutlass.range_constexpr(4):
                        score[4 * nb + k] = acc[k]
            # ---- per-token affine scale/offset, past-length mask, log2 domain
            for nb in cutlass.range_constexpr(4):
                for e in cutlass.range_constexpr(2):
                    token = start + nb * 8 + 2 * col + e
                    k_scale, k_offset = _half2_to_floats(cur_terms[2 * nb + e])
                    for half in cutlass.range_constexpr(2):
                        value = (
                            score[4 * nb + 2 * half + e] * k_scale
                            + k_offset * tally[half]
                        ) * scale_log2
                        if token >= last:
                            value = Float32(-math.inf)
                        score[4 * nb + 2 * half + e] = value
            # ---- online softmax across the row's 4 lanes
            for half in cutlass.range_constexpr(2):
                running = tally[2 + half]
                for nb in cutlass.range_constexpr(4):
                    for e in cutlass.range_constexpr(2):
                        running = cute.arch.fmax(running, score[4 * nb + 2 * half + e])
                running = cute.arch.fmax(
                    running, cute.arch.shuffle_sync_bfly(running, offset=1)
                )
                running = cute.arch.fmax(
                    running, cute.arch.shuffle_sync_bfly(running, offset=2)
                )
                rescale[half] = cute.math.exp2(tally[2 + half] - running, fastmath=True)
                tally[2 + half] = running
                tally[4 + half] = tally[4 + half] * rescale[half]
                tally[6 + half] = tally[6 + half] * rescale[half]
            for slot in cutlass.range_constexpr(16):
                for k in cutlass.range_constexpr(4):
                    out_acc[4 * slot + k] = out_acc[4 * slot + k] * rescale[k // 2]
            # ---- P scaled by the per-token V scale -> A-fragments; the V offsets
            # join the running offset sum
            for kk in cutlass.range_constexpr(2):
                for pair in cutlass.range_constexpr(2):
                    nb = 2 * kk + pair
                    token = start + nb * 8 + 2 * col
                    v_scale0, v_offset0 = _half2_to_floats(cur_v_terms[2 * nb])
                    v_scale1, v_offset1 = _half2_to_floats(cur_v_terms[2 * nb + 1])
                    if (
                        token >= last
                    ):  # unwritten slots may hold garbage terms: zero them
                        v_scale0 = Float32(0.0)
                        v_offset0 = Float32(0.0)
                    if token + 1 >= last:
                        v_scale1 = Float32(0.0)
                        v_offset1 = Float32(0.0)
                    for half in cutlass.range_constexpr(2):
                        p0 = cute.math.exp2(
                            score[4 * nb + 2 * half] - tally[2 + half], fastmath=True
                        )
                        p1 = cute.math.exp2(
                            score[4 * nb + 2 * half + 1] - tally[2 + half],
                            fastmath=True,
                        )
                        tally[4 + half] = tally[4 + half] + p0 + p1
                        tally[6 + half] = (
                            tally[6 + half] + p0 * v_offset0 + p1 * v_offset1
                        )
                        p_frag[4 * kk + half + 2 * pair] = _floats_to_half2(
                            p0 * v_scale0, p1 * v_scale1
                        )
            # ---- O += P V, the V B-fragments straight from the tile's words
            for kk in cutlass.range_constexpr(2):
                for slot in cutlass.range_constexpr(16):
                    b0 = _nibbles(
                        cur_v[8 * kk + slot // 4], slot % 4
                    )  # tokens 2c, 2c+1
                    b1 = _nibbles(
                        cur_v[8 * kk + (16 + slot) // 4], (16 + slot) % 4
                    )  # tokens 2c+8, 2c+9
                    acc = _mma(
                        p_frag[4 * kk],
                        p_frag[4 * kk + 1],
                        p_frag[4 * kk + 2],
                        p_frag[4 * kk + 3],
                        b0,
                        b1,
                        out_acc[4 * slot],
                        out_acc[4 * slot + 1],
                        out_acc[4 * slot + 2],
                        out_acc[4 * slot + 3],
                    )
                    for k in cutlass.range_constexpr(4):
                        out_acc[4 * slot + k] = acc[k]

        # ---- epilogue: reduce denominators and offset sums over the row's lanes,
        # normalize, store the split's partials
        for half in cutlass.range_constexpr(2):
            for neighbor in cutlass.range_constexpr(1, 3):
                tally[4 + half] = tally[4 + half] + cute.arch.shuffle_sync_bfly(
                    tally[4 + half], offset=neighbor
                )
                tally[6 + half] = tally[6 + half] + cute.arch.shuffle_sync_bfly(
                    tally[6 + half], offset=neighbor
                )
        for half in cutlass.range_constexpr(2):
            q_row = row + 8 * half
            if q_row < group:
                inverse = Float32(0.0)
                lse = Float32(-math.inf)
                if tally[4 + half] > Float32(0.0):
                    inverse = Float32(1.0) / tally[4 + half]
                    lse = tally[2 + half] + cute.math.log2(
                        tally[4 + half], fastmath=True
                    )
                if col == 0:
                    m_lses[batch, head, q_row, split] = lse
                for slot in cutlass.range_constexpr(16):
                    for e in cutlass.range_constexpr(2):
                        m_partials[
                            batch, head, q_row, split, 8 * slot + 2 * col + e
                        ] = (
                            out_acc[4 * slot + 2 * half + e] + tally[6 + half]
                        ) * inverse


_COMPILED: dict = {}
_SMS: dict[torch.device, int] = {}


def decode(
    q: torch.Tensor,
    cache: dict,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    out: torch.Tensor | None = None,
    max_ctas: int = 64,
) -> torch.Tensor:
    """q ``[B, HQ, 128]`` (bf16 read as-is, other dtypes cast to fp16) -> out
    ``[B, HQ, 128]`` bf16.

    Graph-capturable: no host syncs; the splits' scratch comes from the caching
    allocator."""
    from vllm.vllm_flash_attn.cute.cute_dsl_utils import to_cute_tensor

    batches, q_heads, _ = q.shape
    kv_heads = cache["k_codes"].shape[1]
    group = q_heads // kv_heads
    assert group <= 16, "one row tile serves at most 16 q heads per kv head"
    page_tokens = cache["k_terms"].shape[2]
    if out is None:
        out = torch.empty(batches, q_heads, DIMS, device=q.device, dtype=torch.bfloat16)
    if batches == 0:
        return out
    if q.device not in _SMS:
        _SMS[q.device] = torch.cuda.get_device_properties(
            q.device
        ).multi_processor_count
    # enough splits for about four CTAs per SM
    ctas = max(1, min(triton.cdiv(4 * _SMS[q.device], batches * kv_heads), max_ctas))
    splits = ctas * 4
    query_bf16 = q.dtype == torch.bfloat16 and q.is_contiguous()
    # bf16 q is read as-is and converted in the prologue: no separate cast kernel
    q32 = (
        q.view(torch.int32)
        if query_bf16
        else q.to(torch.float16).contiguous().view(torch.int32)
    )
    k_codes = cache["k_codes"].view(torch.int32)  # [P, KV_HEADS, BS*16]
    v_codes = cache["v_codes"].view(torch.int32)
    k_terms = (
        cache["k_terms"].view(torch.int32).squeeze(-1)
    )  # [P, KV_HEADS, BS] fp16 pairs
    v_terms = cache["v_terms"].view(torch.int32).squeeze(-1)
    partials = torch.empty(
        batches, kv_heads, 16, splits, DIMS, device=q.device, dtype=torch.float32
    )
    lses = torch.empty(
        batches, kv_heads, 16, splits, device=q.device, dtype=torch.float32
    )
    args = [
        q32,
        k_codes,
        v_codes,
        k_terms,
        v_terms,
        block_table,
        seq_lens,
        partials,
        lses,
    ]
    scale = (1.0 / math.sqrt(DIMS)) * 1.4426950408889634
    key = (group, page_tokens, query_bf16)
    if key not in _COMPILED:
        aligns = [16, 16, 16, 4, 4, 4, 4, 16, 4]
        compiled_args = [
            to_cute_tensor(a, assumed_align=al) for a, al in zip(args, aligns)
        ]
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        _COMPILED[key] = cute.compile(
            DecodeKernel(group, page_tokens, query_bf16),
            *compiled_args,
            Float32(scale),
            Int32(ctas),
            stream,
            options="--enable-tvm-ffi",
        )
    _COMPILED[key](*args, scale, ctas)
    store.combine(partials, lses, out, kv_heads, group)
    return out
