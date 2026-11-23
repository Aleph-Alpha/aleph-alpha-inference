# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""mma.sync routing scans for GPUs without block-scaled FP4 tensor cores.

The tcgen05 scans (`SelectedPoolScan`, `BlockMeanScan` and their prefill
variants) score NVFP4 queries against NVFP4 keys on Blackwell's FP4 tensor
cores. These kernels compute the same scores with m16n8k16 FP16 MMAs: E2M1
values times E4M3 scales are exact in FP16 and their products exact in FP32,
so only the FP32 accumulation order differs. One warp scores eight keys per
MMA group; each lane streams its 16 key bytes and two scales straight from
global memory into B fragments through the software E2M1 conversion. One
kernel serves decode and prefill rows alike; prefill routing is never staged
on these GPUs.

    score[key] = sum_h relu(q_h . k) * 128**-0.5 / 12
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32

from aleph_alpha_inference.kernels.fp16_fragments import (
    dequant_word,
    mma_16816,
    scale_pair,
)
from aleph_alpha_inference.kernels.summary_cache import SummaryCache

SCORE_SCALE = 128**-0.5 / 12
# Tokens per summary block.
BLOCK = SummaryCache.BLOCK
NEG_INF = Float32(-float("inf"))
# Keys per CTA: four warps, each scoring GROUPS groups of eight keys.
GROUPS = 64
KEYS_PER_CTA = 4 * GROUPS * 8


class MmaRoutingScan:
    """Shared kernel: `pool` scores a selected block8 pool, `summary` the
    block-mean summaries of rows whose context exceeds the pool."""

    def __init__(self, mode, pool_tokens=16384, unroll=4):
        if mode not in ("pool", "summary"):
            raise ValueError(f"Unknown routing scan mode {mode!r}")
        self.mode = mode
        self.pool_tokens = pool_tokens
        self.unroll = unroll
        self.key = (mode, pool_tokens, unroll)

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,  # [rows, 16, 64] uint8 NVFP4 routing heads
        qs: cute.Tensor,  # [rows, 16, 2] int32: eight E4M3 scales per head
        k: cute.Tensor,  # [pages, keys per page, 64] uint8
        ks: cute.Tensor,  # [pages, keys per page, 2] int32
        table: cute.Tensor,  # [requests, width] int32
        requests: cute.Tensor,  # [rows] int32
        selected: cute.Tensor,  # [rows, pool blocks] int32 block8 ids; -1 pads
        lengths: cute.Tensor,  # [rows] int32 context length
        scores: cute.Tensor,  # [rows, width] f32
        stream: cuda.CUstream,
    ):
        self.kernel(q, qs, k, ks, table, requests, selected, lengths, scores).launch(
            grid=(cute.ceil_div(scores.shape[1], KEYS_PER_CTA), scores.shape[0], 1),
            block=(128, 1, 1),
            stream=stream,
        )

    @cute.jit
    def load_scale_pair(self, scales, offset, t):
        """Lane t's two E4M3 scale bytes: the t-th 16-bit half of the two
        int32 words at element `offset` of `scales`."""
        halves = cute.make_tensor(
            cute.recast_ptr(scales.iterator + offset, dtype=cutlass.Uint16),
            cute.make_layout((4,)),
        )
        return scale_pair(Uint32(halves[t]))

    @cute.jit
    def load_four_words(self, tensor, offset, words):
        """The 16 bytes at 16-byte aligned byte `offset` into `words`."""
        cute.autovec_copy(
            cute.make_tensor(
                cute.recast_ptr(tensor.iterator + offset, dtype=Uint32),
                cute.make_layout((4,)),
            ),
            words,
        )

    @cute.kernel
    def kernel(  # noqa: C901, PLR0912, PLR0915 -- unrolled GPU loop
        self, q, qs, k, ks, table, requests, selected, lengths, scores
    ):
        tid, _, _ = cute.arch.thread_idx()
        chunk, row, _ = cute.arch.block_idx()
        limit = lengths[row]
        # Summary scores feed pool selection, which only rows past the pool need.
        groups = Int32(GROUPS)
        if cutlass.const_expr(self.mode == "summary"):
            if limit <= self.pool_tokens:
                groups = Int32(0)
        warp = tid // 32
        lane = tid % 32
        g = lane // 4
        t = lane % 4
        request = requests[row]
        keys_per_page = k.shape[1]

        # A fragments: heads g and g + 8 (zero past the 12 routing heads), dims
        # [32t, 32t + 32) in the decode kernel's fragment order: word w of a
        # lane's 16 key bytes feeds MMA k-steps 2w and 2w + 1.
        qa = cute.make_rmem_tensor((2, 16), Uint32)
        for h in cutlass.range_constexpr(2):
            head = g + 8 * h
            words = cute.make_rmem_tensor((4,), Uint32)
            self.load_four_words(
                q,
                cute.assume(
                    Int64(row) * q.stride[0] + head * q.stride[1] + 16 * t, divby=16
                ),
                words,
            )
            qscale = self.load_scale_pair(
                qs, Int64(row) * qs.stride[0] + head * qs.stride[1], t
            )
            for w in cutlass.range_constexpr(4):
                kd = dequant_word(words[w], qscale, w // 2)
                for i in cutlass.range_constexpr(4):
                    qa[h, 4 * w + i] = kd[i]

        base = chunk * KEYS_PER_CTA + warp * GROUPS * 8
        for i in cutlass.range(groups, unroll=self.unroll):
            first = base + 8 * i
            # This lane streams key `first + g`, MMA column g.
            physical = Int32(0)
            slot = Int32(0)
            if cutlass.const_expr(self.mode == "pool"):
                block = selected[row, first // 8]
                token = cutlass.max(block, Int32(0)) * 8 + g
                if block >= 0 and token < limit:
                    physical = table[request, token // keys_per_page]
                    slot = token % keys_per_page
            else:
                key = first + g
                # Complete summaries only, as `BlockMeanScan`: the row's own
                # block may hold later tokens of its prefill chunk, or no
                # summary yet.
                if key < limit // BLOCK:
                    physical = table[request, key // keys_per_page]
                    slot = key % keys_per_page
            raw = cute.make_rmem_tensor((4,), Uint32)
            self.load_four_words(
                k,
                cute.assume(
                    Int64(physical) * k.stride[0] + slot * k.stride[1] + 16 * t,
                    divby=16,
                ),
                raw,
            )
            kscale = self.load_scale_pair(
                ks, Int64(physical) * ks.stride[0] + slot * ks.stride[1], t
            )
            acc = cute.make_rmem_tensor((4,), Float32)
            acc.fill(0)
            for w in cutlass.range_constexpr(4):
                kd = dequant_word(raw[w], kscale, w // 2)
                for h in cutlass.range_constexpr(2):
                    s = 2 * w + h
                    mma_16816(
                        qa[0, 2 * s],
                        qa[1, 2 * s],
                        qa[0, 2 * s + 1],
                        qa[1, 2 * s + 1],
                        kd[2 * h],
                        kd[2 * h + 1],
                        acc,
                    )
            # acc holds (head g, keys 2t and 2t + 1) and (head g + 8, the same).
            v0 = cute.arch.fmax(acc[0], Float32(0))
            v1 = cute.arch.fmax(acc[1], Float32(0))
            if g < 4:
                v0 += cute.arch.fmax(acc[2], Float32(0))
                v1 += cute.arch.fmax(acc[3], Float32(0))
            for offset in (4, 8, 16):
                v0 += cute.arch.shuffle_sync_bfly(v0, offset=offset)
                v1 += cute.arch.shuffle_sync_bfly(v1, offset=offset)
            if g == 0:
                out = cute.make_rmem_tensor((2,), Float32)
                out[0] = v0 * Float32(SCORE_SCALE)
                out[1] = v1 * Float32(SCORE_SCALE)
                column = first + 2 * t
                for j in cutlass.range_constexpr(2):
                    live = cutlass.Boolean(False)
                    if cutlass.const_expr(self.mode == "pool"):
                        block = selected[row, (column + j) // 8]
                        live = block >= 0 and block * 8 + (column + j) % 8 < limit
                    else:
                        live = column + j < limit // BLOCK
                    if not live:
                        out[j] = NEG_INF
                if column < scores.shape[1]:
                    cute.autovec_copy(
                        out,
                        cute.make_tensor(
                            scores.iterator
                            + cute.assume(
                                Int64(row) * scores.stride[0] + column, divby=2
                            ),
                            cute.make_layout((2,)),
                        ),
                    )


class MmaPoolScan(MmaRoutingScan):
    """`SelectedPoolScan` for every row: scores[row, i] scores the token at
    pool slot i, -inf past the row's context or in padding."""

    def __init__(self):
        super().__init__("pool")


class MmaSummaryScan(MmaRoutingScan):
    """`BlockMeanScan`: scores[row, s] scores summary s of rows whose context
    exceeds the pool, -inf for summaries the row's context does not complete;
    other rows are left unwritten."""

    def __init__(self, pool_tokens):
        super().__init__("summary", pool_tokens)
