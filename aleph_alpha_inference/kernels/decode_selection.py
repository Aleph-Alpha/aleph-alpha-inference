# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""Exact CuTe radix selection with ordered bitmap compaction."""
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Uint32


def float_key(value):
    """The bit pattern of `value`. Non-negative floats order like their keys."""
    return value.bitcast(Uint32)


def coarse_key(key, shift: int):
    """The histogram bin of `key`: its value rounded to FP16, bits shifted."""
    half = key.bitcast(Float32).to(cutlass.Float16)
    return half.bitcast(cutlass.Uint16).to(Uint32) >> shift


def warp_sum(value):
    """The sum of `value` over the warp."""
    return cute.arch.warp_redux_sync(value, "add")


def half_value(bits):
    """The value of the FP16 bit pattern in the low half of `bits`."""
    return bits.to(cutlass.Uint16).bitcast(cutlass.Float16).to(Float32)


class RadixSelection:
    """Exact top-k selection per row: one CTA per row (or row segment).

    A histogram over FP16-rounded keys finds the bucket holding the k-th
    largest key, a refinement finds the exact boundary key inside it, and a
    bitmap compaction writes the selected indices in input order. Ties at
    the boundary go to the lowest indices.

    `mode` selects what is ranked and written:
    - 0: summary blocks. `scores[row, b]` ranks block b; `out` gets block IDs.
    - 2: pool tokens. `scores[row, p]` ranks position p of the token pool
      `pool` (8 tokens per block); `out` gets token IDs and `route` the
      attention route (selected, local window, sinks), `used` its length.
    - 3: candidates. `pool[row, c]` holds candidate block IDs ranked by
      `scores[row, block]`; only `ExpandedPoolSelection` builds this mode.

    With `segments` > 1 each row is split into that many CTAs, which select
    from consecutive score ranges of one logical row.

    Pre-conditions:
    - Scores of ranked entries are non-negative and not NaN. Keys with the
      sign bit set are never counted or selected.
    - A row that is selected from has at least `topk` ranked entries. With
      fewer, the boundary is undefined.
    - Mode 0: `scores` has at least ceil(length / block_size) columns per
      row or segment. Mode 3: every candidate block ID is below the score
      width.
    - `threads` is a multiple of 32, at most 1024, and divides the 2048 or
      4096 histogram bins.
    """

    def __init__(
        self,
        mode=0,
        threads=1024,
        segments=1,
        *,
        topk,
        local=128,
        sinks=4,
        block_size=8,
    ):
        self.mode = mode  # 0: block pool, 2: reindex, 3: pool merge.
        self.segments = segments
        self.threads = threads
        self.k = topk
        self.local, self.sinks, self.block_size = local, sinks, block_size
        self.total = self.k + local + sinks

    @cute.jit
    def __call__(
        self,
        scores: cute.Tensor,
        pool: cute.Tensor,
        lengths: cute.Tensor,
        out: cute.Tensor,
        route: cute.Tensor,
        used: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.n = (
            scores.shape[1]
            if self.mode == 0
            else pool.shape[1] * (1 if self.mode == 3 else 8)
        )
        self.iters = cute.ceil_div(self.n, self.threads)
        self.cached = self.n <= 32768
        self.words = cute.ceil_div(self.n, 32)
        self.words_per_thread = cute.ceil_div(self.words, self.threads)
        self.kernel(scores, pool, lengths, out, route, used).launch(
            grid=(scores.shape[0], 1, 1), block=(self.threads, 1, 1), stream=stream
        )

    @cute.jit
    def emit_selected(self, pool, out, row, position, index, block_offset):
        """Write selected input `index` as output `position` of the row."""
        value = Int32(index)
        if cutlass.const_expr(self.mode == 2):
            value = pool[row, index // 8] * 8 + index % 8
        else:
            value += block_offset
        out[row, position] = value

    @cute.jit
    def read_key(self, scores, pool, row, index, length):
        """The key of input `index`, or 0xFFFFFFFF if it is not ranked."""
        key = Uint32(0xFFFFFFFF)
        if cutlass.const_expr(self.mode == 0):
            if index < cute.ceil_div(length, self.block_size):
                key = float_key(scores[row, index])
        elif cutlass.const_expr(self.mode == 3):
            if index < self.n:
                block = pool[row, index]
                if block >= 0 and block < cute.ceil_div(length, self.block_size):
                    key = float_key(scores[row, block])
        else:
            if index < self.n:
                block = pool[row, index // 8]
                token = block * 8 + index % 8
                if block >= 0 and token >= self.sinks and token < length - self.local:
                    key = float_key(scores[row, index])
        return key

    @cute.jit
    def scan(self, value, warp_sums, tid):
        """The sum of `value` over all threads below this one."""
        lane = tid % 32
        warp = tid // 32
        inclusive = value
        for shift in cutlass.range_constexpr(5):
            other = cute.arch.shuffle_sync_up(inclusive, 1 << shift, mask_and_clamp=0)
            if lane >= 1 << shift:
                inclusive += other
        if lane == 31:
            warp_sums[warp] = inclusive
        cute.arch.barrier()
        prior = Uint32(0)
        if lane < warp:
            prior = warp_sums[lane]
        prefix = warp_sum(prior) + inclusive - value
        cute.arch.barrier()
        return prefix

    @cute.jit
    def threshold(
        self, hist, warp_sums, decision, target, tid, bins: cutlass.Constexpr
    ):
        """Find the bin holding the `target`-th largest key of the histogram.

        Returns (bin, keys still to take from that bin, keys in that bin).
        `decision` is shared scratch for the result."""
        local = Uint32(0)
        counts = cute.make_rmem_tensor((cute.ceil_div(bins, self.threads),), Uint32)
        for j in cutlass.range_constexpr(cute.size(counts)):
            slot = bins - 1 - (tid * cute.size(counts) + j)
            counts[j] = Uint32(0)
            if slot >= 0:
                counts[j] = hist[slot]
            local += counts[j]
        prefix = self.scan(local, warp_sums, tid)
        for j in cutlass.range_constexpr(cute.size(counts)):
            count = counts[j]
            if prefix < target and prefix + count >= target:
                decision[0] = bins - 1 - (tid * cute.size(counts) + j)
                decision[1] = target - prefix
                decision[2] = count
            prefix += count
        cute.arch.barrier()
        bucket, remaining, count = decision[0], decision[1], decision[2]
        # Refinement reuses decision[0]. Snapshot all readers before any warp
        # resets that shared slot for the next histogram/compaction stage.
        cute.arch.barrier()
        return bucket, remaining, count

    @cute.kernel
    def kernel(
        self,
        scores: cute.Tensor,
        pool: cute.Tensor,
        lengths: cute.Tensor,
        out: cute.Tensor,
        route: cute.Tensor,
        used: cute.Tensor,
    ):
        sm = cutlass.utils.SmemAllocator()
        self.select(scores, pool, lengths, out, route, used, sm)

    @cute.jit
    def select(self, scores, pool, lengths, out, route, used, sm):
        """Select one row (or row segment) and write its outputs."""
        # Finer bins let more large source pools use the exact boundary sort.
        # Token selection, pool merging and small pools retain the smaller table.
        self.coarse_shift = 3 if self.mode == 0 and self.n >= 16384 else 4
        self.coarse_bins = 32768 >> self.coarse_shift
        self.vectorized = (
            self.cached
            and self.iters % 4 == 0
            and self.n % 4 == 0
            and self.mode != 3
            and scores.stride[0] % 4 == 0
            and scores.stride[1] == 1
        )
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        length = lengths[row // self.segments]
        block_offset = Int32(0)
        if cutlass.const_expr(self.segments > 1):
            block_offset = (row % self.segments) * self.n
            length = cutlass.max(
                Int32(0),
                cutlass.min(
                    length - block_offset * self.block_size,
                    Int32(self.n * self.block_size),
                ),
            )
        hist = sm.allocate_tensor(
            Uint32, cute.make_layout((self.coarse_bins,)), byte_alignment=128
        )
        warp_sums = sm.allocate_tensor(
            Uint32, cute.make_layout((32,)), byte_alignment=128
        )
        decision = sm.allocate_tensor(Uint32, cute.make_layout((3,)), byte_alignment=16)
        above_bits = sm.allocate_tensor(
            Uint32, cute.make_layout((self.words,)), byte_alignment=128
        )
        equal_bits = sm.allocate_tensor(
            Uint32, cute.make_layout((self.words,)), byte_alignment=128
        )
        keys = cute.make_rmem_tensor((self.iters if self.cached else 1,), Uint32)
        run = (
            length > self.total
            if cutlass.const_expr(self.mode == 2)
            else cute.ceil_div(length, self.block_size) > self.k
        )
        if run:
            bucket, remaining, bucket_count = self.coarse_histogram(
                keys, scores, pool, row, length, hist, warp_sums, decision, tid
            )
            lower, upper = self.bucket_interval(bucket)
            lower, remaining, all_selected = self.refine(
                keys,
                scores,
                pool,
                row,
                length,
                hist,
                warp_sums,
                decision,
                tid,
                lower,
                upper,
                remaining,
                bucket_count,
            )
            self.make_bitmaps(
                keys,
                scores,
                pool,
                row,
                length,
                above_bits,
                equal_bits,
                tid,
                lower,
                all_selected,
            )
            selected, output_prefix = self.compact_ties(
                above_bits, equal_bits, warp_sums, tid, remaining
            )
            self.write_selected(
                selected, out, pool, row, tid, block_offset, output_prefix
            )
        else:
            self.copy_short_pool(out, row, tid, length, block_offset)
        self.format_route(out, route, used, row, tid, length)

    @cute.jit
    def read_vector(self, scores, pool, row, index):
        """Four consecutive scores starting at `index`, with their pool
        block and first token ID (modes other than 0)."""
        block = Int32(0)
        token = Int32(index)
        if cutlass.const_expr(self.mode != 0):
            if index < self.n:
                block = pool[row, index // 8]
            token = block * 8 + index % 8
        values = cute.make_rmem_tensor((4,), Float32)
        values.fill(Float32(-float("inf")))
        valid = index < self.n and block >= 0
        if valid:
            src = cute.make_tensor(
                scores.iterator + row * scores.stride[0] + cute.assume(index, divby=4),
                cute.make_layout((4,)),
            )
            cute.autovec_copy(src, values)
        return block, token, values

    @cute.jit
    def coarse_histogram(
        self, keys, scores, pool, row, length, hist, warp_sums, decision, tid
    ):
        """Count every key in its FP16 bin and cache the keys if they fit.

        Returns `threshold`'s result for the k-th largest key."""
        if cutlass.const_expr(self.vectorized):
            for j in cutlass.range_constexpr(
                cute.ceil_div(self.coarse_bins, self.threads)
            ):
                hist[j * self.threads + tid] = 0
            cute.arch.barrier()
            for i in cutlass.range_constexpr(self.iters // 4):
                index = (i * self.threads + tid) * 4
                block, token, values = self.read_vector(scores, pool, row, index)
                for j in cutlass.range_constexpr(4):
                    valid = index + j < self.n
                    if cutlass.const_expr(self.mode == 0):
                        valid = valid and index + j < cute.ceil_div(
                            length, self.block_size
                        )
                    else:
                        valid = (
                            valid
                            and block >= 0
                            and token + j >= self.sinks
                            and token + j < length - self.local
                        )
                    key = float_key(values[j]) if valid else Uint32(0xFFFFFFFF)
                    keys[i * 4 + j] = key
                    self.count_coarse(hist, key)
            cute.arch.barrier()
            result = self.threshold(
                hist, warp_sums, decision, Uint32(self.k), tid, self.coarse_bins
            )
        else:
            result = self.scalar_histogram(
                keys, scores, pool, row, length, hist, warp_sums, decision, tid
            )
        return result

    @cute.jit
    def make_bitmaps(
        self,
        keys,
        scores,
        pool,
        row,
        length,
        above_bits,
        equal_bits,
        tid,
        lower,
        all_selected,
    ):
        """Write one bit per input: `above_bits` for keys that are selected
        outright, `equal_bits` for keys tied at the boundary `lower`."""
        if cutlass.const_expr(self.vectorized):
            # Refined FP32 keys lie in [0, +inf], so this cannot overflow.
            # Signed comparison also rejects the invalid 0xffffffff sentinel.
            cutoff = Int32(lower) + Int32(not all_selected)
            lane = tid % 32
            warp = tid // 32
            for i in cutlass.range_constexpr(self.iters // 4):
                above = Uint32(0)
                equal = Uint32(0)
                for j in cutlass.range_constexpr(4):
                    key = keys[i * 4 + j]
                    keep = Int32(key) >= cutoff
                    tie = not all_selected and key == lower
                    if keep:
                        above |= Uint32(1) << j
                    if tie:
                        equal |= Uint32(1) << j
                # Eight lanes contribute four consecutive input bits each.
                # Restore input order before stable bitmap compaction.
                above <<= (lane % 8) * 4
                equal <<= (lane % 8) * 4
                above = cute.arch.warp_reduction_sum(above, threads_in_group=8)
                equal = cute.arch.warp_reduction_sum(equal, threads_in_group=8)
                word = i * (self.threads // 8) + warp * 4 + lane // 8
                if lane % 8 == 0 and word < self.words:
                    above_bits[word] = above
                    equal_bits[word] = equal
            cute.arch.barrier()
        else:
            self.scalar_bitmaps(
                keys,
                scores,
                pool,
                row,
                length,
                above_bits,
                equal_bits,
                tid,
                lower,
                all_selected,
            )

    @cute.jit
    def count_coarse(self, hist, key):
        """Count a non-negative `key` in its FP16 bin."""
        if key < Uint32(0x80000000):
            cute.arch.atomic_add(
                hist.iterator + coarse_key(key, self.coarse_shift),
                Uint32(1),
                scope="cta",
            )

    @cute.jit
    def count_digit(self, hist, key, lower, upper, shift):
        """Count a `key` in [lower, upper] by its 10-bit digit at `shift`."""
        if key >= lower and key <= upper:
            cute.arch.atomic_add(
                hist.iterator + ((key >> shift) & 1023), Uint32(1), scope="cta"
            )

    @cute.jit
    def scalar_histogram(
        self, keys, scores, pool, row, length, hist, warp_sums, decision, tid
    ):
        """`coarse_histogram` reading one key per thread and step."""
        for j in cutlass.range_constexpr(cute.ceil_div(self.coarse_bins, self.threads)):
            hist[j * self.threads + tid] = 0
        cute.arch.barrier()
        if cutlass.const_expr(self.cached):
            for i in cutlass.range_constexpr(self.iters):
                key = self.read_key(scores, pool, row, i * self.threads + tid, length)
                keys[i] = key
                self.count_coarse(hist, key)
        else:
            # Streamed keys: the loop body lives in a helper because the DSL
            # treats a name bound in the constexpr branch above as carried
            # into this dynamic loop (and rejects it as None on entry).
            for i in cutlass.range(self.iters, unroll=1):
                self.count_coarse(
                    hist,
                    self.read_key(scores, pool, row, i * self.threads + tid, length),
                )
        cute.arch.barrier()
        bucket, remaining, bucket_count = self.threshold(
            hist, warp_sums, decision, Uint32(self.k), tid, self.coarse_bins
        )
        return bucket, remaining, bucket_count

    @cute.jit
    def bucket_interval(self, bucket):
        """The smallest and largest FP32 key that fall into FP16 bin `bucket`."""
        # RN-even FP16 boundaries give an exact FP32 key interval. The
        # first half in each bin is even, and the last is odd.
        width = 1 << self.coarse_shift
        overflow_bin = 0x7C00 >> self.coarse_shift
        lower = Uint32(0)
        upper = Uint32(0x7F800000)
        if bucket > 0 and bucket < overflow_bin:
            lower = float_key(
                (half_value(bucket * width) + half_value(bucket * width - 1))
                * Float32(0.5)
            )
        if bucket < overflow_bin - 1:
            upper = float_key(
                (
                    half_value(bucket * width + width - 1)
                    + half_value(bucket * width + width)
                )
                * Float32(0.5)
            ) - Uint32(1)
        elif bucket == overflow_bin - 1:
            upper = float_key(Float32(65520)) - Uint32(1)
        elif bucket == overflow_bin:
            lower = float_key(Float32(65520))
        return lower, upper

    @cute.jit
    def refine(
        self,
        keys,
        scores,
        pool,
        row,
        length,
        hist,
        warp_sums,
        decision,
        tid,
        lower,
        upper,
        remaining,
        bucket_count,
    ):
        """Narrow the key interval [lower, upper] of the boundary bucket.

        `remaining` keys of the bucket's `bucket_count` are still to select.
        Returns (boundary key, keys to take at or from the boundary, whether
        every key from the boundary key up is selected)."""
        all_selected = bucket_count == remaining
        # The exact histogram count guarantees that a bucket of at most 128
        # keys fits the scratch. Larger buckets and streamed keys take the
        # full FP32 radix refinement.
        small = False
        if cutlass.const_expr(self.cached):
            small = bucket_count <= 128 and not all_selected
        if small:
            lower, remaining = self.small_refine(
                keys,
                hist,
                decision,
                tid,
                lower,
                upper,
                remaining,
            )
            all_selected = False
        else:
            lower, remaining, all_selected = self.radix_refine(
                keys,
                scores,
                pool,
                row,
                length,
                hist,
                warp_sums,
                decision,
                tid,
                lower,
                upper,
                remaining,
                all_selected,
            )
        return lower, remaining, all_selected

    @cute.jit
    def small_refine(
        self,
        keys,
        scratch,
        decision,
        tid,
        lower,
        upper,
        remaining,
    ):
        """Find the boundary among at most 128 cached keys in [lower, upper]
        by sorting them in `scratch`.

        Returns (boundary key, number of keys equal to it to select)."""
        if tid < 128:
            scratch[tid] = 0
        if tid == 0:
            decision[0] = 0
        cute.arch.barrier()
        for i in cutlass.range_constexpr(self.iters):
            key = keys[i]
            if key >= lower and key <= upper:
                pos = cute.arch.atomic_add(decision.iterator, Uint32(1), scope="cta")
                scratch[pos] = key
        cute.arch.barrier()
        if tid < 32:
            self.select_boundary_width(scratch, decision, remaining, tid, 4)
        cute.arch.barrier()
        return decision[0], decision[1]

    @cute.jit
    def select_boundary_width(
        self, scratch, decision, remaining, tid, per_lane: cutlass.Constexpr
    ):
        """Sort the `32 * per_lane` keys in `scratch` within one warp and
        store the `remaining`-th largest and the number of keys equal to it
        to select in `decision[0]` and `decision[1]`."""
        values = cute.make_rmem_tensor((per_lane,), Uint32)
        for j in cutlass.range_constexpr(per_lane):
            values[j] = scratch[tid * per_lane + j]
        self.sort_boundary_values(values, tid)
        target = 32 * per_lane - Int32(remaining)
        pivot = Uint32(0)
        for j in cutlass.range_constexpr(per_lane):
            if target % per_lane == j:
                pivot = values[j]
        pivot = cute.arch.shuffle_sync(pivot, target // per_lane)
        greater = Uint32(0)
        for j in cutlass.range_constexpr(per_lane):
            if values[j] > pivot:
                greater += 1
        above = warp_sum(greater)
        if tid == 0:
            decision[0] = pivot
            decision[1] = remaining - above

    @cute.jit
    def sort_boundary_values(self, values, tid):
        """Sort the warp's keys ascending by (lane, slot) with a bitonic
        network; `values` are this lane's slots."""
        per_lane = cutlass.const_expr(cute.size(values))
        log_values = cutlass.const_expr(per_lane.bit_length() - 1)
        update = cute.make_rmem_tensor((per_lane,), Uint32)
        for stage in cutlass.range_constexpr(1, 6 + log_values):
            for bit in cutlass.range_constexpr(stage - 1, -1, -1):
                for j in cutlass.range_constexpr(per_lane):
                    other = Uint32(0)
                    if cutlass.const_expr(bit < log_values):
                        other = values[j ^ (1 << bit)]
                    else:
                        other = cute.arch.shuffle_sync_bfly(
                            values[j], 1 << (bit - log_values)
                        )
                    index = tid * per_lane + j
                    ascending = ((index & (1 << stage)) == 0) == (
                        (index & (1 << bit)) == 0
                    )
                    update[j] = (
                        cutlass.min(values[j], other)
                        if ascending
                        else cutlass.max(values[j], other)
                    )
                values.store(update.load())

    @cute.jit
    def radix_refine(
        self,
        keys,
        scores,
        pool,
        row,
        length,
        hist,
        warp_sums,
        decision,
        tid,
        lower,
        upper,
        remaining,
        all_selected,
    ):
        """Narrow [lower, upper] ten key bits per pass until one key value
        is left or every key in the interval is selected.

        Returns `refine`'s result."""
        while lower < upper and not all_selected:
            shift = cutlass.max(
                Int32(0), Int32(22) - Int32(cute.arch.clz(lower ^ upper))
            )
            base = lower & (Uint32(0xFFFFFFFF) << (shift + 10))
            for j in cutlass.range_constexpr(cute.ceil_div(1024, self.threads)):
                index = j * self.threads + tid
                if index < 1024:
                    hist[index] = 0
            cute.arch.barrier()
            if cutlass.const_expr(self.cached):
                for i in cutlass.range_constexpr(self.iters):
                    self.count_digit(hist, keys[i], lower, upper, shift)
            else:
                for i in cutlass.range(self.iters, unroll=1):
                    self.count_digit(
                        hist,
                        self.read_key(
                            scores, pool, row, i * self.threads + tid, length
                        ),
                        lower,
                        upper,
                        shift,
                    )
            cute.arch.barrier()
            digit, remaining, count = self.threshold(
                hist, warp_sums, decision, remaining, tid, 1024
            )
            digit_lower = base | (digit << shift)
            lower = cutlass.max(lower, digit_lower)
            upper = cutlass.min(upper, digit_lower | ((Uint32(1) << shift) - 1))
            all_selected = count == remaining
            cute.arch.barrier()
        return lower, remaining, all_selected

    @cute.jit
    def scalar_bitmaps(
        self,
        keys,
        scores,
        pool,
        row,
        length,
        above_bits,
        equal_bits,
        tid,
        lower,
        all_selected,
    ):
        """`make_bitmaps` with one key per thread and step."""
        cutoff = Int32(lower) + Int32(not all_selected)
        lane = tid % 32
        warp = tid // 32
        # One bit per original input retains the order without sorting IDs.
        if cutlass.const_expr(self.cached):
            for i in cutlass.range_constexpr(self.iters):
                self.ballot_word(
                    keys[i],
                    above_bits,
                    equal_bits,
                    i * (self.threads // 32) + warp,
                    lane,
                    cutoff,
                    lower,
                    all_selected,
                )
        else:
            for i in cutlass.range(self.iters, unroll=1):
                self.ballot_word(
                    self.read_key(scores, pool, row, i * self.threads + tid, length),
                    above_bits,
                    equal_bits,
                    i * (self.threads // 32) + warp,
                    lane,
                    cutoff,
                    lower,
                    all_selected,
                )
        cute.arch.barrier()

    @cute.jit
    def ballot_word(
        self, key, above_bits, equal_bits, word, lane, cutoff, lower, all_selected
    ):
        """Write the warp's 32 selected and tied bits as bitmap word `word`."""
        keep = Int32(key) >= cutoff
        equal = (not all_selected) and key == lower
        a = Uint32(cute.arch.vote_ballot_sync(keep))
        e = Uint32(cute.arch.vote_ballot_sync(equal))
        if lane == 0 and word < self.words:
            above_bits[word] = a
            equal_bits[word] = e

    @cute.jit
    def compact_ties(self, above_bits, equal_bits, warp_sums, tid, remaining):
        """Add the first `remaining` tied inputs to the selected ones.

        Returns this thread's selected bitmap words and the number of
        selected inputs before them."""
        selected = cute.make_rmem_tensor((self.words_per_thread,), Uint32)
        ties = cute.make_rmem_tensor((self.words_per_thread,), Uint32)
        local_equal = Uint32(0)
        for j in cutlass.range_constexpr(self.words_per_thread):
            word = tid * self.words_per_thread + j
            selected[j] = Uint32(0)
            ties[j] = Uint32(0)
            if word < self.words:
                selected[j] = above_bits[word]
                ties[j] = equal_bits[word]
            local_equal += cute.arch.popc(ties[j])
        equal_prefix = self.scan(local_equal, warp_sums, tid)
        local_count = Uint32(0)
        for j in cutlass.range_constexpr(self.words_per_thread):
            bits = ties[j]
            count = cute.arch.popc(bits)
            allowed = Int32(remaining) - Int32(equal_prefix)
            trimmed = bits
            if allowed <= 0:
                trimmed = Uint32(0)
            elif allowed < Int32(count):
                trimmed = Uint32(0)
                for bit in cutlass.range(allowed, unroll=1):
                    low_bit = bits & (Uint32(0) - bits)
                    trimmed |= low_bit
                    bits ^= low_bit
            selected[j] |= trimmed
            equal_prefix += count
            local_count += cute.arch.popc(selected[j])
        output_prefix = self.scan(local_count, warp_sums, tid)
        return selected, output_prefix

    @cute.jit
    def write_selected(
        self, selected, out, pool, row, tid, block_offset, output_prefix
    ):
        """Write this thread's selected inputs, in input order, starting at
        output `output_prefix`."""
        for j in cutlass.range_constexpr(self.words_per_thread):
            bits = selected[j]
            while bits != 0:
                bit = Uint32(0) - bits
                bit = bits & bit
                index = (
                    (tid * self.words_per_thread + j) * 32
                    + 31
                    - Int32(cute.arch.clz(bit))
                )
                self.emit_selected(pool, out, row, output_prefix, index, block_offset)
                output_prefix += 1
                bits ^= bit

    @cute.jit
    def copy_short_pool(self, out, row, tid, length, block_offset):
        """Rows with at most k blocks select every block: write their IDs
        and pad with -1."""
        if cutlass.const_expr(self.mode == 0):
            for i in cutlass.range_constexpr(cute.ceil_div(self.k, self.threads)):
                index = i * self.threads + tid
                if index < self.k:
                    out[row, index] = (
                        index + block_offset
                        if index < cute.ceil_div(length, self.block_size)
                        else -1
                    )

    @cute.jit
    def format_route(self, out, route, used, row, tid, length):
        """Mode 2: write the row's attention route (selected tokens, local
        window, sinks; all tokens for short rows) and its length."""
        if cutlass.const_expr(self.mode == 2):
            cute.arch.barrier()
            for i in cutlass.range_constexpr(cute.ceil_div(self.total, self.threads)):
                index = i * self.threads + tid
                if index < self.total:
                    value = Int32(0)
                    if length <= self.total:
                        value = index if index < length else 0
                    elif index < self.k:
                        value = out[row, index]
                    elif index < self.k + self.local:
                        value = length - self.local + index - self.k
                    else:
                        value = index - self.k - self.local
                    route[row, index] = value
            if tid == 0:
                used[row] = cutlass.min(length, self.total)


class ExpandedPoolSelection(RadixSelection):
    """Summary-block pool selection that also writes the expanded token pool.

    Called as (scores, small, hi, small, pool, lo): besides the selected
    summary blocks in `small`, pool[r, f p + e] = f small[r, p] + e for
    f = pool width / small width, or -1 for empty slots. That is exactly what
    the separate expand_pool kernels wrote, without their launch. `lo` is
    not read.
    """

    def __init__(self, topk, block_size, merge=False):
        # `merge` selects among per-segment candidates (mode 3) instead.
        super().__init__(
            3 if merge else 0, threads=512, topk=topk, block_size=block_size
        )

    @cute.jit
    def write_expanded(self, out, route, row, position, value):
        """Write the token blocks of summary block `value` to the pool."""
        factor = cutlass.const_expr(route.shape[1] // out.shape[1])
        base = value * factor
        for e in cutlass.range_constexpr(factor):
            token_block = Int32(-1)
            if value >= 0:
                token_block = base + e
            route[row, position * factor + e] = token_block

    @cute.jit
    def emit_selected(self, pool, out, row, position, index, block_offset):
        value = Int32(index) + block_offset
        if cutlass.const_expr(self.mode == 3):
            value = pool[row, index]
        out[row, position] = value
        self.write_expanded(out, self.route_out, row, position, value)

    @cute.jit
    def copy_short_pool(self, out, row, tid, length, block_offset):
        for i in cutlass.range_constexpr(cute.ceil_div(self.k, self.threads)):
            index = i * self.threads + tid
            if index < self.k:
                value = (
                    index + block_offset
                    if index < cute.ceil_div(length, self.block_size)
                    else Int32(-1)
                )
                out[row, index] = value
                self.write_expanded(out, self.route_out, row, index, value)

    @cute.kernel
    def kernel(
        self,
        scores: cute.Tensor,
        pool: cute.Tensor,
        lengths: cute.Tensor,
        out: cute.Tensor,
        route: cute.Tensor,
        used: cute.Tensor,
    ):
        self.route_out = route
        sm = cutlass.utils.SmemAllocator()
        self.select(scores, pool, lengths, out, route, used, sm)
