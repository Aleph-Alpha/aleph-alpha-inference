# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""Inline-PTX helpers for FP16 mma.sync fragments over NVFP4 operands.

Blackwell converts E2M1 pairs with `cvt.rn.f16x2.e2m1x2`. Other GPUs build
the same FP16 pairs with prmt lookups: the FP16 encodings of the eight E2M1
magnitudes (0, .5, 1, 1.5, 2, 3, 4, 6) have a zero low byte, so one prmt
looks up four high bytes, a second spreads two of them into an f16x2 word,
and a sign-replicating prmt plus lop3 merge the sign bits. Both are exact.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import arith, llvm, nvvm
from cutlass.cutlass_dsl import dsl_user_op, T

from aleph_alpha_inference.kernels.arch import is_blackwell

# FP16 high bytes of the E2M1 magnitudes, indexed by the magnitude bits.
_F16_HIGH_BYTES = (0x3E3C3800, 0x46444240)
# lop3 immediate for f(a, b, c) = (a & ~c) | (b & c).
_MERGE = 0xD8
# Rounding boundaries between the E2M1 magnitudes as FP32 hex literals. Ties
# go to the even-mantissa neighbour, so every other boundary is inclusive.
_E2M1_BOUNDARIES = (
    ("gt", "0f3E800000"),  # 0.25
    ("ge", "0f3F400000"),  # 0.75
    ("gt", "0f3FA00000"),  # 1.25
    ("ge", "0f3FE00000"),  # 1.75
    ("gt", "0f40200000"),  # 2.5
    ("ge", "0f40600000"),  # 3.5
    ("gt", "0f40A00000"),  # 5
)


def inline_asm(ret, args, body, constraints, loc, ip):
    """Emit side-effect free inline PTX over the DSL values `args`."""
    return llvm.inline_asm(
        ret,
        [a.ir_value(loc=loc, ip=ip) for a in args],
        body,
        constraints,
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _dequant_body(scale):
    """PTX for four f16x2 words from the eight nibbles of $4, times `scale`.

    Word i holds (low nibble, high nibble) of byte i, as cvt.e2m1x2 orders them.
    """
    if is_blackwell():
        body = "{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3},$4; "
        for i in range(4):
            body += f"cvt.rn.f16x2.e2m1x2 ${i},b{i}; mul.rn.f16x2 ${i},${i},{scale}; "
        return body + "}"
    low, high = _F16_HIGH_BYTES
    body = (
        "{ .reg .b32 sel,sel2,w4,d0,d1,sp,m,z,c,ll,lh; "
        f"mov.b32 ll,{low:#x}; mov.b32 lh,{high:#x}; mov.b32 c,0x80008000; "
        "mov.b32 z,0; and.b32 sel,$4,0x77777777; shl.b32 w4,$4,4; "
        "shr.b32 sel2,sel,16; prmt.b32 d0,ll,lh,sel; prmt.b32 d1,ll,lh,sel2; "
    )
    for i in range(4):
        # Bytes i of w4 and of $4 carry the signs of nibble pair i.
        signs = ((8 | i) << 4) | ((12 | i) << 12)
        body += (
            f"prmt.b32 sp,d{i // 2},z,{0x1404 if i % 2 == 0 else 0x3424:#x}; "
            f"prmt.b32 m,w4,$4,{signs:#x}; lop3.b32 ${i},sp,m,c,{_MERGE:#x}; "
            f"mul.rn.f16x2 ${i},${i},{scale}; "
        )
    return body + "}"


@dsl_user_op
def mma_16816(a0, a1, a2, a3, b0, b1, c, *, loc=None, ip=None):
    """Add the FP16 product of fragments `a` and `b` to the FP32 fragment `c`."""
    values = inline_asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [a0, a1, a2, a3, b0, b1, c[0], c[1], c[2], c[3]],
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{$0,$1,$2,$3},{$4,$5,$6,$7},{$8,$9},{$10,$11,$12,$13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,0,1,2,3",
        loc,
        ip,
    )
    for i in range(4):
        c[i] = Float32(llvm.extractvalue(T.f32(), values, [i], loc=loc, ip=ip))


@dsl_user_op
def scale_pair(bytes2: Uint32, *, loc=None, ip=None):
    """Two E4M3 scales (low two bytes) as an exact FP16 pair."""
    low = arith.trunci(T.i16(), bytes2.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    pair = nvvm.convert_f8x2_to_f16x2(
        ir.VectorType.get([2], T.f16()),
        llvm.bitcast(ir.VectorType.get([2], T.i8()), low, loc=loc, ip=ip),
        ir.TypeAttr.get(cutlass.Float8E4M3FN.mlir_type),
        loc=loc,
        ip=ip,
    )
    return Uint32(llvm.bitcast(T.i32(), pair, loc=loc, ip=ip))


@dsl_user_op
def dequant_word(
    word: Uint32,
    scales: Uint32,
    half: cutlass.Constexpr = None,
    *,
    loc=None,
    ip=None,
):
    """Four e2m1x2 bytes times a packed scale pair, as exact FP16 pairs.

    With `half` 0 or 1, both elements of each pair use that half of `scales`.
    """
    if half is None:
        body = _dequant_body("$5")
    else:
        pick = "{l,l}" if half == 0 else "{h,h}"
        body = "{ .reg .b16 l,h; .reg .b32 s; mov.b32 {l,h},$5; "
        body += f"mov.b32 s,{pick}; " + _dequant_body("s") + " }"
    values = inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [word, scales],
        body,
        "=r,=r,=r,=r,r,r",
        loc,
        ip,
    )
    return [
        Uint32(llvm.extractvalue(T.i32(), values, [i], loc=loc, ip=ip))
        for i in range(4)
    ]


@dsl_user_op
def pack_pair(lo: Float32, hi: Float32, *, loc=None, ip=None):
    """Two FP32 values rounded to an FP16 pair, `lo` in the low half."""
    pair = nvvm.convert_f32x2_to_f16x2(
        ir.VectorType.get([2], T.f16()),
        hi.ir_value(loc=loc, ip=ip),
        lo.ir_value(loc=loc, ip=ip),
        rnd=nvvm.FPRoundingMode.RN,
        loc=loc,
        ip=ip,
    )
    return Uint32(llvm.bitcast(T.i32(), pair, loc=loc, ip=ip))


@dsl_user_op
def bf16_to_f16x2(word: Uint32, *, loc=None, ip=None):
    """A BF16 pair converted to an FP16 pair."""
    body = (
        "{ .reg .b16 l,h; .reg .f32 f0,f1; mov.b32 {l,h},$1; "
        "cvt.f32.bf16 f0,l; cvt.f32.bf16 f1,h; cvt.rn.f16x2.f32 $0,f1,f0; }"
    )
    return Uint32(inline_asm(T.i32(), [word], body, "=r,r", loc, ip))


@dsl_user_op
def f16x2_to_bf16x2(word: Uint32, *, loc=None, ip=None):
    """An FP16 pair converted to a BF16 pair, rounding to nearest even."""
    body = (
        "{ .reg .b16 l,h; .reg .f32 f0,f1; mov.b32 {l,h},$1; "
        "cvt.f32.f16 f0,l; cvt.f32.f16 f1,h; cvt.rn.bf16x2.f32 $0,f1,f0; }"
    )
    return Uint32(inline_asm(T.i32(), [word], body, "=r,r", loc, ip))


@dsl_user_op
def fp4_pair(x: Float32, y: Float32, *, loc=None, ip=None):
    """Round `x` and `y` to nearest-even FP4 (E2M1), saturating; `x` is the
    low nibble of the returned byte."""
    if is_blackwell():
        return cutlass.Uint8(
            nvvm.convert_f32x2_to_f4x2(
                y.ir_value(loc=loc, ip=ip),
                x.ir_value(loc=loc, ip=ip),
                ir.TypeAttr.get(cutlass.Float4E2M1FN.mlir_type),
                rnd=nvvm.FPRoundingMode.RN,
                loc=loc,
                ip=ip,
            )
        )
    # Magnitude code = number of rounding boundaries below |x|.
    body = "{ .reg .pred p; .reg .f32 a; .reg .b32 c,s; mov.b32 $0,0; "
    for i, arg in enumerate(("$1", "$2")):
        body += f"abs.f32 a,{arg}; mov.b32 c,0; "
        for cmp, value in _E2M1_BOUNDARIES:
            body += f"setp.{cmp}.f32 p,a,{value}; @p add.u32 c,c,1; "
        body += f"mov.b32 s,{arg}; shr.u32 s,s,28; and.b32 s,s,8; or.b32 c,c,s; "
        body += f"shl.b32 c,c,{4 * i}; or.b32 $0,$0,c; " if i else "or.b32 $0,$0,c; "
    body += "}"
    return cutlass.Uint8(Uint32(inline_asm(T.i32(), [x, y], body, "=r,f,f", loc, ip)))


@cute.jit
def fma_f32x2(a, b, c):
    """Two FP32 fused multiply-adds, one packed instruction on Blackwell."""
    if cutlass.const_expr(is_blackwell()):
        return cute.arch.fma_packed_f32x2(a, b, c)
    return (a[0] * b[0] + c[0], a[1] * b[1] + c[1])


@cute.jit
def mul_f32x2(a, b):
    """Two FP32 multiplies, one packed instruction on Blackwell."""
    if cutlass.const_expr(is_blackwell()):
        return cute.arch.mul_packed_f32x2(a, b)
    return (a[0] * b[0], a[1] * b[1])
