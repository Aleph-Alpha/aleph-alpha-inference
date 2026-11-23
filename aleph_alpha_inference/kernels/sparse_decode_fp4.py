# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""Register-fragment sparse decode over paged NVFP4 KV.

One CTA owns a (request row, token split) pair and its four warps own the four
KV heads. Per 32-token step the CTA gathers each token's 256-byte K and V rows
(all four heads) and their scales with 16-byte cp.async copies, 16 lanes per
row, into a swizzled shared ring (L2 evict-first: the KV stream is not
reused). Each warp then dequantises its head straight into mma.sync m16n8k16
B fragments:

- QK^T: the head dimension is permuted (consistently for Q and K) so that each
  thread's K fragment is one contiguous 16-byte run of its token. The same
  bytes feed `cvt.rn.f16x2.e2m1x2`, with no shared-memory round trip.
- PV: each thread owns 16 contiguous output dims (one NVFP4 scale group); a
  nibble interleave pairs adjacent tokens so a single e2m1x2 conversion yields
  the token-pair register the B fragment wants.

Splits are a run-time scalar, so one executable serves every batch size. A
single split writes the BF16 output directly; otherwise FP32 partials and
log2-domain (max, sum) statistics go to the caller's scratch for
`SplitCombine`.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op, T

from aleph_alpha_inference.kernels.fp16_fragments import (
    bf16_to_f16x2,
    dequant_word,
    fma_f32x2,
    inline_asm,
    mma_16816,
    mul_f32x2,
    pack_pair,
    scale_pair,
)

# 1/sqrt(128) * log2(e): scores live in the log2 domain.
SCORE_SCALE = 0.08838834764831845 * 1.4426950408889634
STEP = 32
# Ring stages: one step is computed while the next one loads. Two stages leave
# room for MIN_BLOCKS CTAs (12 warps) per SM.
STAGES = 2
MIN_BLOCKS = 3
# Unrolling two steps stops ptxas permuting the 64 accumulators on every back
# edge.
UNROLL = 2
NEG_INF = Float32(-float("inf"))
# A stage holds STEP tokens of all four KV heads: K rows (256 B), V rows, then
# 32 bytes of K scales and of V scales per token.
V_BASE = STEP * 256
KS_BASE = 2 * STEP * 256
VS_BASE = KS_BASE + STEP * 32
STAGE_BYTES = VS_BASE + STEP * 32


@dsl_user_op
def evict_first_policy(*, loc=None, ip=None):
    """L2 policy for the KV stream: keep reused data (partials, routes) cached."""
    return Int64(
        inline_asm(
            T.i64(),
            [],
            "createpolicy.fractional.L2::evict_first.b64 $0, 1.0;",
            "=l",
            loc,
            ip,
        )
    )


@dsl_user_op
def cp_async_hint(dst, src, policy: Int64, *, loc=None, ip=None):
    """cp.async of 16 bytes with an L2 cache-policy hint."""
    llvm.inline_asm(
        None,
        [
            Int32(dst.toint(loc=loc, ip=ip)).ir_value(loc=loc, ip=ip),
            src.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            policy.ir_value(loc=loc, ip=ip),
        ],
        "cp.async.cg.shared.global.L2::cache_hint [$0], [$1], 16, $2;",
        "r,l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


# Swizzles (as in nvfp4_attention.cu) keep the 16-lane row copies and every
# fragment read free of bank conflicts.
def k_offset(token, chunk):
    """Rows 2m and 2m + 1, read together, sit in opposite bank halves."""
    return 256 * token + 16 * (chunk ^ ((token & 1) << 2))


def v_offset(token, chunk):
    """Rows 2t (t = 0..3), read together, take distinct 32-byte bank quarters."""
    return V_BASE + 256 * token + 16 * (chunk ^ (((token >> 1) & 3) << 1))


def scale_offset(token, byte):
    """Rows r and r + 4 swap 16-byte halves."""
    return 32 * token + (byte ^ (((token >> 2) & 1) << 4))


class SparseDecodeFP4Split:
    """Attention of each decode row over `total` routed tokens, in `splits`
    token ranges per row.

    With one split the kernel writes the BF16 output. Otherwise it writes FP32
    partial outputs and their (max, sum) reductions for `SplitCombine`.
    """

    def __init__(self, total):
        self.total = total
        self.map_tokens = (total + STEP - 1) // STEP * STEP

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        ks: cute.Tensor,
        vs: cute.Tensor,
        table: cute.Tensor,
        route: cute.Tensor,
        used: cute.Tensor,
        req: cute.Tensor,
        partial: cute.Tensor,
        reductions: cute.Tensor,
        out: cute.Tensor,
        splits: Int32,
        stream: cuda.CUstream,
    ):
        smem = self.map_tokens * 4 + STAGES * STAGE_BYTES + 4 * 4096 + 512
        self.kernel(
            q, k, v, ks, vs, table, route, used, req, partial, reductions, out, splits
        ).launch(
            grid=(splits, q.shape[0], 1),
            block=(128, 1, 1),
            smem=smem,
            stream=stream,
            min_blocks_per_mp=MIN_BLOCKS,
        )

    @cute.kernel
    def kernel(  # noqa: C901, PLR0912, PLR0915 -- unrolled GPU loop
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        ks: cute.Tensor,
        vs: cute.Tensor,
        table: cute.Tensor,
        route: cute.Tensor,
        used: cute.Tensor,
        req: cute.Tensor,
        partial: cute.Tensor,
        reductions: cute.Tensor,
        out: cute.Tensor,
        splits: Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        split, row, _ = cute.arch.block_idx()
        lane = tid % 32
        head = tid // 32
        g = lane // 4
        t = lane % 4
        page_size = k.shape[1]

        sm = cutlass.utils.SmemAllocator()
        token_map = sm.allocate_tensor(
            Uint32, cute.make_layout((self.map_tokens,)), byte_alignment=16
        )
        ring = sm.allocate_tensor(
            cutlass.Uint8,
            cute.make_layout((STAGES * STAGE_BYTES,)),
            byte_alignment=128,
        )
        q_words = sm.allocate_tensor(
            Uint32, cute.make_layout((4 * 1024,)), byte_alignment=16
        )

        count = used[row]
        chunk = (cute.ceil_div(Int32(self.total), splits) + STEP - 1) // STEP * STEP
        lo = split * chunk
        hi = cutlass.min(lo + chunk, Int32(self.total))
        limit = cutlass.min(hi, count)
        span = cutlass.max(hi - lo, Int32(0))
        steps = cute.ceil_div(span, STEP)

        # Physical token ids of this split; masked tokens read live token 0.
        request = req[row]
        for i in cutlass.range(tid, steps * STEP, 128, unroll=1):
            n = lo + i
            key = Int32(0)
            if n < limit:
                key = route[row, n]
            page = Int32(0)
            if count > 0:
                page = table[request, key // page_size]
            token_map[i] = Uint32(page) * Uint32(page_size) + Uint32(key % page_size)

        # Q fragments, head dims permuted as the K fragments are, stored in
        # A-fragment order per (chunk, lane) so each MMA loads one 16-byte word.
        qw = cute.make_rmem_tensor((2, 16), Uint32)
        for r in cutlass.range_constexpr(2):
            m = g + 8 * r
            if m < 12:
                src = cute.make_tensor(
                    q.iterator
                    + cute.assume(
                        row * q.stride[0] + (head * 12 + m) * 128 + 32 * t, divby=8
                    ),
                    cute.make_layout((32,)),
                )
                values = cute.make_rmem_tensor((32,), BFloat16)
                cute.autovec_copy(src, values)
                words = cute.recast_tensor(values, Uint32)
                for i in cutlass.range_constexpr(16):
                    qw[r, i] = bf16_to_f16x2(words[i])
            else:
                for i in cutlass.range_constexpr(16):
                    qw[r, i] = Uint32(0)
        for c in cutlass.range_constexpr(8):
            frag = cute.make_rmem_tensor((4,), Uint32)
            frag[0] = qw[0, 2 * c]
            frag[1] = qw[1, 2 * c]
            frag[2] = qw[0, 2 * c + 1]
            frag[3] = qw[1, 2 * c + 1]
            cute.autovec_copy(
                frag,
                cute.make_tensor(
                    q_words.iterator
                    + cute.assume(((head * 8 + c) * 32 + lane) * 4, divby=4),
                    cute.make_layout((4,)),
                ),
            )
        cute.arch.barrier()

        policy = evict_first_policy()

        # Ring prologue: stage 0 receives step 0.
        if steps > 0:
            self.issue(k, v, ks, vs, token_map, ring, 0, 0, tid, policy)
        cute.arch.cp_async_commit_group()

        acc = cute.make_rmem_tensor((16, 4), Float32)
        acc.fill(0)
        stats = cute.make_rmem_tensor((4,), Float32)  # max g, max g+8, sum g, sum g+8
        stats[0] = NEG_INF
        stats[1] = NEG_INF
        stats[2] = Float32(0)
        stats[3] = Float32(0)

        for step in cutlass.range(steps, unroll=UNROLL):
            # Every thread's copies of this step have landed, and every warp is
            # done with the stage the refill below overwrites.
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()
            ahead = step + 1
            if ahead < steps:
                self.issue(
                    k, v, ks, vs, token_map, ring, ahead, ahead % STAGES, tid, policy
                )
            cute.arch.cp_async_commit_group()
            stage = (step % STAGES) * STAGE_BYTES
            self.compute(
                ring,
                stage,
                q_words,
                head,
                lane,
                acc,
                stats,
                lo + step * STEP,
                limit,
                g,
                t,
            )
        cute.arch.cp_async_wait_group(0)

        for r in cutlass.range_constexpr(2):
            total = stats[2 + r]
            total += cute.arch.shuffle_sync_bfly(total, offset=1)
            total += cute.arch.shuffle_sync_bfly(total, offset=2)
            stats[2 + r] = total
        self.store(partial, reductions, out, acc, stats, splits, split, row, head, g, t)

    @cute.jit
    def issue(self, k, v, ks, vs, token_map, ring, step, slot, tid, policy):
        """All 128 threads gather one stage: 16 lanes per 256-byte K or V row."""
        stage = slot * STAGE_BYTES
        page_size = k.shape[1]
        lane = tid % 32
        chunk = lane % 16
        for j in cutlass.range_constexpr(4):
            token = 8 * j + 2 * (tid // 32) + lane // 16
            phys = token_map[step * STEP + token]
            page = Int64(phys // Uint32(page_size))
            slot_id = phys % Uint32(page_size)
            for kind in cutlass.range_constexpr(2):
                data = v if kind else k
                src = (
                    data.iterator
                    + page * data.stride[0]
                    + slot_id * data.stride[1]
                    + chunk * 16
                )
                offset = v_offset(token, chunk) if kind else k_offset(token, chunk)
                cp_async_hint(
                    ring.iterator + cute.assume(stage + offset, divby=16),
                    src,
                    policy,
                )
        # Scales: threads 0-63 copy K scales, 64-127 V scales, 16 bytes each.
        token = tid % 64 // 2
        half = tid % 2
        phys = token_map[step * STEP + token]
        page = Int64(phys // Uint32(page_size))
        slot_id = phys % Uint32(page_size)
        offset = stage + scale_offset(token, 16 * half)
        if tid < 64:
            cp_async_hint(
                ring.iterator + cute.assume(KS_BASE + offset, divby=16),
                ks.iterator + page * ks.stride[0] + slot_id * ks.stride[1] + 16 * half,
                policy,
            )
        else:
            cp_async_hint(
                ring.iterator + cute.assume(VS_BASE + offset, divby=16),
                vs.iterator + page * vs.stride[0] + slot_id * vs.stride[1] + 16 * half,
                policy,
            )

    @cute.jit
    def compute(  # noqa: C901, PLR0912, PLR0915 -- unrolled GPU step
        self, ring, stage, q_words, head, lane, acc, stats, first, limit, g, t
    ):
        words = cute.recast_ptr(ring.iterator, dtype=Uint32)
        # QK^T: four independent n-block accumulation chains, interleaved.
        raw = cute.make_rmem_tensor((4, 4), Uint32)
        kscale = cute.make_rmem_tensor((4,), Uint32)
        for j in cutlass.range_constexpr(4):
            token = 8 * j + g
            vec = cute.make_rmem_tensor((4,), Uint32)
            cute.autovec_copy(
                cute.make_tensor(
                    words
                    + cute.assume(
                        (stage + k_offset(token, 4 * head + t)) // 4, divby=4
                    ),
                    cute.make_layout((4,)),
                ),
                vec,
            )
            for w in cutlass.range_constexpr(4):
                raw[j, w] = vec[w]
            pair = cute.make_tensor(
                cute.recast_ptr(
                    ring.iterator
                    + cute.assume(
                        stage + KS_BASE + scale_offset(token, 8 * head + 2 * t), divby=2
                    ),
                    dtype=cutlass.Uint16,
                ),
                cute.make_layout((1,)),
            )[0]
            kscale[j] = scale_pair(Uint32(pair))
        scores = cute.make_rmem_tensor((4, 4), Float32)
        scores.fill(0)
        for w in cutlass.range_constexpr(4):
            a = cute.make_rmem_tensor((2, 4), Uint32)
            for h in cutlass.range_constexpr(2):
                c = 2 * w + h
                cute.autovec_copy(
                    cute.make_tensor(
                        q_words.iterator
                        + cute.assume(((head * 8 + c) * 32 + lane) * 4, divby=4),
                        cute.make_layout((4,)),
                    ),
                    a[h, None],
                )
            for j in cutlass.range_constexpr(4):
                kd = dequant_word(raw[j, w], kscale[j], w // 2)
                frag = cute.make_rmem_tensor((4,), Float32)
                for e in cutlass.range_constexpr(4):
                    frag[e] = scores[j, e]
                for h in cutlass.range_constexpr(2):
                    mma_16816(
                        a[h, 0],
                        a[h, 1],
                        a[h, 2],
                        a[h, 3],
                        kd[2 * h],
                        kd[2 * h + 1],
                        frag,
                    )
                for e in cutlass.range_constexpr(4):
                    scores[j, e] = frag[e]

        # Online softmax in the log2 domain; masked tokens score -inf. Scores
        # stay unscaled until the exponent, which folds the scale into an FMA.
        if first + STEP > limit:
            for j in cutlass.range_constexpr(4):
                for e in cutlass.range_constexpr(4):
                    if first + 8 * j + 2 * t + e % 2 >= limit:
                        scores[j, e] = NEG_INF
        for r in cutlass.range_constexpr(2):
            peak = cute.arch.fmax(scores[0, 2 * r], scores[0, 2 * r + 1])
            for j in cutlass.range_constexpr(1, 4):
                peak = cute.arch.fmax(
                    peak, cute.arch.fmax(scores[j, 2 * r], scores[j, 2 * r + 1])
                )
            peak = cute.arch.warp_reduction_max(peak, threads_in_group=4)
            fresh = cute.arch.fmax(peak * Float32(SCORE_SCALE), stats[r])
            safe = fresh if fresh > NEG_INF else Float32(0)
            alpha = cute.math.exp2(stats[r] - safe, fastmath=True)
            stats[r] = fresh
            total = Float32(0)
            for j in cutlass.range_constexpr(4):
                x0, x1 = fma_f32x2(
                    (scores[j, 2 * r], scores[j, 2 * r + 1]),
                    (Float32(SCORE_SCALE), Float32(SCORE_SCALE)),
                    (-safe, -safe),
                )
                p0 = cute.math.exp2(x0, fastmath=True)
                p1 = cute.math.exp2(x1, fastmath=True)
                scores[j, 2 * r] = p0
                scores[j, 2 * r + 1] = p1
                total += p0 + p1
            stats[2 + r] = stats[2 + r] * alpha + total
            for nb in cutlass.range_constexpr(16):
                acc[nb, 2 * r], acc[nb, 2 * r + 1] = mul_f32x2(
                    (acc[nb, 2 * r], acc[nb, 2 * r + 1]), (alpha, alpha)
                )

        for kc in cutlass.range_constexpr(2):
            a0 = pack_pair(scores[2 * kc, 0], scores[2 * kc, 1])
            a1 = pack_pair(scores[2 * kc, 2], scores[2 * kc, 3])
            a2 = pack_pair(scores[2 * kc + 1, 0], scores[2 * kc + 1, 1])
            a3 = pack_pair(scores[2 * kc + 1, 2], scores[2 * kc + 1, 3])
            vraw = cute.make_rmem_tensor((4, 2), Uint32)
            sc = cute.make_rmem_tensor((4,), Uint32)
            for i in cutlass.range_constexpr(4):
                token = 16 * kc + 2 * t + (i % 2) + 8 * (i // 2)
                vec = cute.make_rmem_tensor((2,), Uint32)
                cute.autovec_copy(
                    cute.make_tensor(
                        words
                        + cute.assume(
                            (stage + v_offset(token, 4 * head + g // 2) + 8 * (g % 2))
                            // 4,
                            divby=2,
                        ),
                        cute.make_layout((2,)),
                    ),
                    vec,
                )
                vraw[i, 0] = vec[0]
                vraw[i, 1] = vec[1]
                sc[i] = Uint32(
                    cute.make_tensor(
                        ring.iterator
                        + stage
                        + VS_BASE
                        + scale_offset(token, 8 * head + g),
                        cute.make_layout((1,)),
                    )[0]
                )
            s01 = scale_pair(sc[0] | (sc[1] << 8))
            s23 = scale_pair(sc[2] | (sc[3] << 8))
            for half in cutlass.range_constexpr(2):
                w0, w1 = vraw[0, half], vraw[1, half]
                w2, w3 = vraw[2, half], vraw[3, half]
                low = Uint32(0x0F0F0F0F)
                high = Uint32(0xF0F0F0F0)
                even01 = dequant_word((w0 & low) | ((w1 << 4) & high), s01)
                odd01 = dequant_word(((w0 >> 4) & low) | (w1 & high), s01)
                even23 = dequant_word((w2 & low) | ((w3 << 4) & high), s23)
                odd23 = dequant_word(((w2 >> 4) & low) | (w3 & high), s23)
                for i in cutlass.range_constexpr(4):
                    for parity in cutlass.range_constexpr(2):
                        nb = half * 8 + 2 * i + parity
                        frag = cute.make_rmem_tensor((4,), Float32)
                        for e in cutlass.range_constexpr(4):
                            frag[e] = acc[nb, e]
                        b0 = odd01[i] if parity else even01[i]
                        b1 = odd23[i] if parity else even23[i]
                        mma_16816(a0, a1, a2, a3, b0, b1, frag)
                        for e in cutlass.range_constexpr(4):
                            acc[nb, e] = frag[e]

    @cute.jit
    def store(
        self, partial, reductions, out, acc, stats, splits, split, row, head, g, t
    ):
        """Write a split's result: the normalised BF16 output if it is the only
        split, else its FP32 partial output and (max, sum) reductions."""
        for r in cutlass.range_constexpr(2):
            m = g + 8 * r
            if m < 12:
                total = stats[2 + r]
                if splits == 1:
                    inverse = Float32(0)
                    if total > Float32(0):
                        inverse = Float32(1) / total
                    values = cute.make_rmem_tensor((32,), BFloat16)
                    for i in cutlass.range_constexpr(32):
                        values[i] = BFloat16(acc[i % 16, 2 * r + i // 16] * inverse)
                    dst = cute.make_tensor(
                        out.iterator
                        + cute.assume(
                            row * out.stride[0] + (head * 12 + m) * 128 + 32 * t,
                            divby=8,
                        ),
                        cute.make_layout((32,)),
                    )
                    cute.autovec_copy(values, dst)
                else:
                    values = cute.make_rmem_tensor((32,), Float32)
                    for i in cutlass.range_constexpr(32):
                        values[i] = acc[i % 16, 2 * r + i // 16]
                    slot = row * splits + split
                    dst = cute.make_tensor(
                        partial.iterator
                        + cute.assume(
                            Int64(slot) * partial.stride[0]
                            + m * partial.stride[1]
                            + head * partial.stride[2]
                            + 32 * t,
                            divby=4,
                        ),
                        cute.make_layout((32,)),
                    )
                    cute.autovec_copy(values, dst)
                    if t == 0:
                        pair = cute.make_rmem_tensor((2,), Float32)
                        pair[0] = stats[r]
                        pair[1] = total
                        cute.autovec_copy(
                            pair,
                            cute.make_tensor(
                                reductions.iterator
                                + cute.assume(
                                    Int64(slot) * reductions.stride[0]
                                    + head * reductions.stride[1]
                                    + m * 2,
                                    divby=2,
                                ),
                                cute.make_layout((2,)),
                            ),
                        )


class SplitCombine:
    """Merge up to 64 run-time split partials of `SparseDecodeFP4Split` into BF16."""

    @cute.jit
    def __call__(
        self,
        partial: cute.Tensor,
        reductions: cute.Tensor,
        out: cute.Tensor,
        splits: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(partial, reductions, out, splits).launch(
            grid=(out.shape[0] * 12, 1, 1), block=(128, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(  # noqa: C901 -- unrolled GPU reduction
        self,
        partial: cute.Tensor,
        reductions: cute.Tensor,
        out: cute.Tensor,
        splits: Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        h = bid * 4 + tid // 32
        row = h // 48
        head = h % 48
        qh = head % 12
        kv = head // 12
        lane = tid % 32
        # Lane s holds the statistics of splits s and s + 32, read in parallel.
        peaks = cute.make_rmem_tensor((2,), Float32)
        sums = cute.make_rmem_tensor((2,), Float32)
        peak = NEG_INF
        for i in cutlass.range_constexpr(2):
            peaks[i] = NEG_INF
            sums[i] = Float32(0)
            s = lane + 32 * i
            if s < splits:
                stat = cute.make_rmem_tensor((2,), Float32)
                cute.autovec_copy(
                    cute.make_tensor(
                        reductions.iterator
                        + cute.assume(
                            Int64(row * splits + s) * reductions.stride[0]
                            + kv * reductions.stride[1]
                            + qh * 2,
                            divby=2,
                        ),
                        cute.make_layout((2,)),
                    ),
                    stat,
                )
                peaks[i] = stat[0]
                sums[i] = stat[1]
            peak = cute.arch.fmax(peak, peaks[i])
        peak = cute.arch.warp_reduction_max(peak)
        total = Float32(0)
        weights = cute.make_rmem_tensor((2,), Float32)
        for i in cutlass.range_constexpr(2):
            # Empty splits carry -inf maxima and may hold non-finite partials.
            weights[i] = Float32(0)
            if peaks[i] > NEG_INF:
                weights[i] = cute.math.exp2(peaks[i] - peak, fastmath=True)
            total += weights[i] * sums[i]
        total = cute.arch.warp_reduction_sum(total)
        acc = cute.make_rmem_tensor((4,), Float32)
        acc.fill(0)
        for s in cutlass.range(splits, unroll=4):
            mine = weights[0]
            if s >= 32:
                mine = weights[1]
            weight = cute.arch.shuffle_sync(mine, s % 32)
            if weight > Float32(0):
                vals = cute.make_rmem_tensor((4,), Float32)
                cute.autovec_copy(
                    cute.make_tensor(
                        partial.iterator
                        + cute.assume(
                            Int64(row * splits + s) * partial.stride[0]
                            + qh * partial.stride[1]
                            + kv * partial.stride[2]
                            + lane * 4,
                            divby=4,
                        ),
                        cute.make_layout((4,)),
                    ),
                    vals,
                )
                for j in cutlass.range_constexpr(4):
                    acc[j] += weight * vals[j]
        inverse = Float32(0)
        if total > Float32(0):
            inverse = Float32(1) / total
        result = cute.make_rmem_tensor((4,), BFloat16)
        for j in cutlass.range_constexpr(4):
            result[j] = BFloat16(acc[j] * inverse)
        cute.autovec_copy(
            result,
            cute.make_tensor(
                out.iterator
                + cute.assume(row * out.stride[0] + head * 128 + lane * 4, divby=4),
                cute.make_layout((4,)),
            ),
        )
