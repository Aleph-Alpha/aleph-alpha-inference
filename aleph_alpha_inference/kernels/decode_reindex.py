# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""CuTe block-scaled FP4 scoring of a selected block8 pool."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass import Float32, Int32, Uint32
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.utils import blackwell_helpers as bw, blockscaled_layout as bs


class SelectedPoolScan:
    """Score each query's 16k-token pool: grid (pool chunk, query), 128 threads.

    Every CTA first stages its block IDs, physical pages and per-tile
    liveness in shared memory (`stage_index`), keeping index loads off each
    tile's path. It then scores one 128-token tile per pipeline step and
    skips the MMA of tiles without a live block. Rows with at most
    `short_tokens` tokens are not scored at all.

    Pre-condition: `scores` has `selected.shape[1] * 8` columns. The scan
    bounds its tiles by the pool width and writes scores unguarded.

    Subclasses define what a CTA stages (`stage_index`, `tile_live`), where a
    tile's keys are (`locate`), how many score columns a row has
    (`scan_length`) and what is published (`store_scores`).
    """

    def __init__(self, short_tokens=0, pages=8, stages=2):
        # NVFP4 routing keys: one E4M3 scale per 16 values.
        self.sf_vec = 16
        self.sf_words = 2
        self.sf_dtype = cutlass.Float8E4M3FN
        self.pages = pages
        self.stages = stages
        # Rows with at most this many tokens are skipped: another scan scores
        # them (`GroupedShortScan`) or nothing reads their scores.
        self.short_tokens = short_tokens
        # Query heads per KV head and their dimension.
        self.heads = 12
        self.head_dim = 128
        # Head columns per MMA: the packed query pads `heads` to 16.
        self.query_width = 16
        self.tile = (128, self.query_width, self.head_dim)
        # The 16-column accumulator and scale factors fit 32 columns, so
        # TMEM never limits occupancy.
        self.tmem_columns = 32

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        qs: cute.Tensor,
        k: cute.Tensor,
        ks: cute.Tensor,
        table: cute.Tensor,
        requests: cute.Tensor,
        selected: cute.Tensor,
        lengths: cute.Tensor,
        scores: cute.Tensor,
        stream: cuda.CUstream,
    ):
        mma, tb, al, bl, sal, sbl = self.query_mma(q)
        self.kernel(
            mma,
            tb,
            qs,
            k,
            ks,
            table,
            requests,
            selected,
            lengths,
            scores,
            al,
            bl,
            sal,
            sbl,
        ).launch(
            grid=(cute.ceil_div(scores.shape[1], self.pages * 128), q.shape[0], 1),
            block=(128, 1, 1),
            stream=stream,
        )

    @cute.jit
    def query_tensor(self, q):
        """The packed queries as the MMA's B operand, one query per tile."""
        return cute.make_tensor(
            cute.recast_ptr(q.iterator, dtype=cutlass.Float4E2M1FN),
            cute.make_layout((16, 128, q.shape[0]), stride=(128, 1, 2048)),
        )

    @cute.jit
    def query_mma(self, q):
        """The tiled MMA, the query's TMA atom and the shared memory layouts
        of keys, queries and their scale factors."""
        fp4 = cutlass.Float4E2M1FN
        mma = cute.make_tiled_mma(
            tcgen05.MmaMXF4NVF4Op(
                self.sf_dtype,
                (128, self.query_width, 64),
                tcgen05.CtaGroup.ONE,
                tcgen05.OperandSource.SMEM,
            )
        )
        al = bw.make_smem_layout_a(mma, self.tile, fp4, self.stages)
        bl = bw.make_smem_layout_b(mma, self.tile, fp4, 1)
        sal = bs.make_smem_layout_sfa(mma, self.tile, self.sf_vec, self.stages)
        sbl = bs.make_smem_layout_sfb(mma, self.tile, self.sf_vec, 1)
        qb = self.query_tensor(q)
        op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        cluster = cute.make_layout((1, 1, 1, 1))
        tb = cute.nvgpu.make_tiled_tma_atom_B(
            op,
            qb,
            cute.slice_(bl, (None, None, None, 0)),
            self.tile,
            mma,
            cluster.shape,
        )
        return mma, tb, al, bl, sal, sbl

    @cute.kernel
    def kernel(
        self,
        mma: cute.TiledMma,
        tb: cpasync.TmaInfo,
        qs: cute.Tensor,
        k: cute.Tensor,
        ks: cute.Tensor,
        table: cute.Tensor,
        requests: cute.Tensor,
        selected: cute.Tensor,
        lengths: cute.Tensor,
        scores: cute.Tensor,
        al: cute.ComposedLayout,
        bl: cute.ComposedLayout,
        sal: cute.Layout,
        sbl: cute.Layout,
    ):
        """One CTA: score `pages` tiles of query `row`, starting at tile
        `chunk * pages`."""
        tid, _, _ = cute.arch.thread_idx()
        chunk, row, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        sm = cutlass.utils.SmemAllocator()
        index, live, hi = self.stage_index(
            sm, k, table, selected, requests, lengths, row, chunk, tid
        )
        if hi <= self.short_tokens:
            # This scan does not score the row.
            pass
        elif live:
            self.score(
                index,
                mma,
                tb,
                qs,
                k,
                ks,
                table,
                requests,
                selected,
                hi,
                scores,
                al,
                bl,
                sal,
                sbl,
                sm,
                tid,
                chunk,
                row,
                warp,
            )
        else:
            # Short contexts leave most of the 16k pool empty: publish -inf
            # without reserving TMEM or loading the query.
            for p in cutlass.range_constexpr(self.pages):
                token = (chunk * self.pages + p) * 128 + tid
                if token < scores.shape[1]:
                    scores[row, token] = Float32(-float("inf"))

    @cute.jit
    def score(
        self,
        index,
        mma,
        tb,
        qs,
        k,
        ks,
        table,
        requests,
        selected,
        hi,
        scores,
        al,
        bl,
        sal,
        sbl,
        sm,
        tid,
        chunk,
        row,
        warp,
    ):
        """Set up shared memory, barriers and tensor memory, load the query
        and scan this CTA's tiles. `hi` is the row's end."""
        sa = sm.allocate_tensor(cutlass.Float4E2M1FN, al.outer, 128, swizzle=al.inner)
        sb = sm.allocate_tensor(cutlass.Float4E2M1FN, bl.outer, 128, swizzle=bl.inner)
        sfa = sm.allocate_tensor(self.sf_dtype, sal, 128)
        sfb = sm.allocate_tensor(self.sf_dtype, sbl, 128)
        # Barrier 0 completes the query load, barrier 1 each tile's MMA.
        bars = sm.allocate_tensor(
            cutlass.Uint64, cute.make_layout((2,)), byte_alignment=8
        )
        holder = sm.allocate_tensor(Int32, cute.make_layout((1,)), byte_alignment=4)
        for i in cutlass.range_constexpr(2):
            if tid == 0:
                cute.arch.mbarrier_init(bars.iterator + i, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()
        allocator = cutlass.utils.TmemAllocator(
            holder.iterator, barrier_for_retrieve=pipeline.NamedBarrier(1, 128)
        )
        pool = allocator.reserve(self.tmem_columns)
        allocator.relinquish_alloc_permit()
        acc_shape = mma.partition_shape_C((128, self.query_width))
        acc = pool.allocate_tensor(mma.make_fragment_C(acc_shape).layout, Float32)
        sfat, sfbt = self.scale_tmem(pool, mma, sal, sbl)
        thr = mma.get_slice(0)
        ar = mma.make_fragment_A(sa)
        br = mma.make_fragment_B(sb)
        s2ta, sfas, sfad = self.scale_copy(sfa, sfat)
        s2tb, sfbs, sfbd = self.scale_copy(sfb, sfbt)
        # Each thread loads its key's `query_width` (16) head columns.
        load = cute.make_copy_atom(
            tcgen05.Ld32x32bOp(tcgen05.Repetition.x16, tcgen05.Pack.NONE), Float32
        )
        tcopy = tcgen05.make_tmem_copy(load, acc)
        tthr = tcopy.get_slice(tid)
        ts = tthr.partition_S(acc)
        coords = tthr.partition_D(
            thr.partition_C(cute.make_identity_tensor((128, self.query_width)))
        )
        values = cute.make_rmem_tensor(coords.shape, Float32)
        sfaw = cute.make_tensor(
            cute.recast_ptr(sfa.iterator, dtype=Uint32),
            cute.make_layout((self.stages * 128 * self.sf_words,)),
        )
        self.load_query(tb, thr, sb, sfb, qs, row, tid, warp, bars, s2tb, sfbs, sfbd)
        self.scan(
            index,
            k,
            ks,
            table,
            requests,
            selected,
            hi,
            scores,
            row,
            chunk,
            tid,
            warp,
            sa,
            al.inner,
            sfaw,
            bars,
            s2ta,
            sfas,
            sfad,
            mma,
            acc,
            ar,
            sfat,
            br,
            sfbt,
            tcopy,
            ts,
            values,
            coords,
        )
        cute.arch.barrier()
        allocator.free(pool.base_ptr)

    @cute.jit
    def stage_index(self, sm, k, table, selected, requests, lengths, row, chunk, tid):
        """Stage this CTA's block IDs and pages in shared memory; return
        (index, any block live, row end).

        index[0] holds the block IDs, index[1] their physical pages and
        index[2, t] the count of tile t's live blocks: a block is live when
        it exists and starts before the row's end. The row's bounds and block
        IDs load together, so the CTA waits on two global round trips."""
        index = sm.allocate_tensor(
            Int32,
            cute.make_layout((3, self.pages * 16), stride=(self.pages * 16, 1)),
            byte_alignment=16,
        )
        hi = lengths[row]
        request = requests[row]
        for i in cutlass.range_constexpr(cute.ceil_div(self.pages * 16, 128)):
            j = i * 128 + tid
            column = chunk * self.pages * 16 + j
            block = Int32(-1)
            if j < self.pages * 16 and column < selected.shape[1]:
                block = selected[row, column]
            token = cutlass.max(block, 0) * 8
            alive = block >= 0 and token < hi and hi > self.short_tokens
            physical = Int32(0)
            if alive:
                physical = table[request, token // k.shape[1]]
            # Lanes 16 h .. 16 h + 15 hold tile j // 16 of this warp.
            ballot = Uint32(cute.arch.vote_ballot_sync(alive))
            lane = tid % 32
            if j < self.pages * 16:
                index[0, j] = block
                index[1, j] = physical
                if lane % 16 == 0:
                    index[2, j // 16] = Int32(
                        cute.arch.popc((ballot >> (lane & 16)) & Uint32(0xFFFF))
                    )
        cute.arch.barrier()
        live = Int32(0)
        for t in cutlass.range_constexpr(self.pages):
            live += index[2, t]
        return index, live > 0, hi

    @cute.jit
    def tile_live(self, index, p):
        """Whether tile `p` of this CTA has a block to score."""
        return index[2, p] > 0

    @cute.jit
    def locate(self, index, table, request, k, hi, local, logical, m):
        """Key row `m` of a tile: its physical page, its slot in the page and
        whether it exists. `local` and `logical` are the tile's index in this
        CTA and in the row."""
        j = local * 16 + m // 8
        block = index[0, j]
        token = Uint32(cutlass.max(block, 0) * 8 + m % 8)
        valid_token = block >= 0 and token < hi
        physical = index[1, j]
        slot = token % k.shape[1]
        if not valid_token:
            physical = Int32(0)
            slot = Uint32(0)
        return physical, slot, valid_token

    @cute.jit
    def load_key(
        self,
        index,
        table,
        request,
        k,
        ks,
        hi,
        local,
        logical,
        sa,
        swizzle,
        sfaw,
        stage,
        tid,
    ):
        """Copy one tile's keys and key scales into stage `stage`. `swizzle`
        is the swizzle of the key stages `sa`."""
        atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            cutlass.Uint8,
            num_bits_per_copy=128,
        )
        for i in cutlass.range_constexpr(4):
            m = i * 32 + tid // 4
            d = (tid % 4) * 32
            physical, slot, valid_token = self.locate(
                index, table, request, k, hi, local, logical, m
            )
            offset = cute.assume(
                cutlass.Int64(physical) * k.stride[0] + slot * k.stride[1] + d // 2,
                divby=16,
            )
            src = cute.make_tensor(k.iterator + offset, cute.make_layout((16,)))
            shared_offset = cute.assume(
                cute.crd2idx(((m, d % 64), 0, d // 64, stage), sa.layout), divby=32
            )
            dst = cute.make_tensor(
                cute.recast_ptr(
                    sa.iterator + shared_offset,
                    swizzle_=swizzle,
                    dtype=cutlass.Uint8,
                ),
                cute.make_layout((16,)),
            )
            # Do not gather missing rows from physical token zero: that
            # serializes large batches on the same cache lines. Predicated
            # cp.async zero-fills shared memory for them instead.
            valid = cute.make_rmem_tensor(
                cute.make_layout((16,), stride=(0,)), cutlass.Boolean
            )
            valid[0] = valid_token
            cute.copy(atom, src, dst, pred=valid)
        physical, slot, valid_token = self.locate(
            index, table, request, k, hi, local, logical, (tid % 4) * 32 + tid // 4
        )
        self.load_key_scales(ks, sfaw, stage, tid, physical, slot, valid_token)

    @cute.jit
    def load_key_scales(self, ks, sfaw, stage, position, physical, slot, valid_token):
        """Copy one key's scale words to `position` of stage `stage`, or
        zeros unless `valid_token`."""
        atom_sf = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.ALWAYS),
            Uint32,
            num_bits_per_copy=32,
        )
        valid = cute.make_rmem_tensor((1,), cutlass.Boolean)
        valid[0] = valid_token
        for half in cutlass.range_constexpr(self.sf_words):
            src = cute.make_tensor(
                cute.recast_ptr(ks.iterator, dtype=Uint32)
                + cutlass.Int64(physical) * ks.stride[0]
                + slot * ks.stride[1]
                + half,
                cute.make_layout((1,)),
            )
            dst = cute.make_tensor(
                sfaw.iterator + stage * 128 * self.sf_words + half * 128 + position,
                cute.make_layout((1,)),
            )
            cute.copy(atom_sf, src, dst, pred=valid)

    @cute.jit
    def scan(
        self,
        index,
        k,
        ks,
        table,
        requests,
        selected,
        hi,
        scores,
        row,
        chunk,
        tid,
        warp,
        sa,
        swizzle,
        sfaw,
        bars,
        s2ta,
        sfas,
        sfad,
        mma,
        acc,
        ar,
        sfat,
        br,
        sfbt,
        tcopy,
        ts,
        values,
        coords,
    ):
        """Score this CTA's tiles through a pipeline of `stages` key stages:
        each step starts the copy of a later tile's keys, then scores the
        tile whose keys have arrived."""
        phase = Int32(0)
        request = requests[row]
        length = self.scan_length(hi, selected, scores)
        first_page = chunk * self.pages
        for stage in cutlass.range_constexpr(self.stages - 1):
            if stage < self.pages and (first_page + stage) * 128 < length:
                self.load_key(
                    index,
                    table,
                    request,
                    k,
                    ks,
                    hi,
                    Int32(stage),
                    first_page + stage,
                    sa,
                    swizzle,
                    sfaw,
                    Int32(stage),
                    tid,
                )
            cute.arch.cp_async_commit_group()
        for p in cutlass.range(self.pages, unroll=1):
            logical_page = first_page + p
            stage = p % self.stages
            if logical_page * 128 < length:
                if (
                    p + self.stages - 1 < self.pages
                    and (logical_page + self.stages - 1) * 128 < length
                ):
                    self.load_key(
                        index,
                        table,
                        request,
                        k,
                        ks,
                        hi,
                        p + self.stages - 1,
                        logical_page + self.stages - 1,
                        sa,
                        swizzle,
                        sfaw,
                        (p + self.stages - 1) % self.stages,
                        tid,
                    )
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(self.stages - 1)
                cute.arch.barrier()
                if self.tile_live(index, p):
                    self.score_tile(
                        index,
                        p,
                        logical_page,
                        stage,
                        phase,
                        scores,
                        row,
                        hi,
                        warp,
                        bars,
                        s2ta,
                        sfas,
                        sfad,
                        mma,
                        acc,
                        ar,
                        sfat,
                        br,
                        sfbt,
                        tcopy,
                        ts,
                        values,
                        coords,
                    )
                    phase ^= 1
                else:
                    # No live block: skip the MMA and publish -inf. `scores`
                    # is as wide as the pool (see the class pre-condition).
                    scores[row, logical_page * 128 + tid] = Float32(-float("inf"))
                cute.arch.barrier()

    @cute.jit
    def scan_length(self, hi, selected, scores):
        """The number of score columns this row's scan covers."""
        return selected.shape[1] * 8

    @cute.jit
    def score_tile(
        self,
        index,
        p,
        logical_page,
        stage,
        phase,
        scores,
        row,
        hi,
        warp,
        bars,
        s2ta,
        sfas,
        sfad,
        mma,
        acc,
        ar,
        sfat,
        br,
        sfbt,
        tcopy,
        ts,
        values,
        coords,
    ):
        """Score the tile staged in `stage` and publish its scores."""
        cute.arch.fence_view_async_shared()
        if warp == 0:
            cute.copy(s2ta, sfas[(None, None, None, None, stage)], sfad)
            mma.set(tcgen05.Field.ACCUMULATE, False)
            cute.gemm(
                mma,
                acc,
                [ar[(None, None, None, stage)], sfat],
                [br[(None, None, None, 0)], sfbt],
                acc,
            )
            with cute.arch.elect_one():
                tcgen05.commit(bars.iterator + 1)
        cute.arch.mbarrier_wait(bars.iterator + 1, phase)
        cute.copy(tcopy, ts, values)
        cute.arch.fence_view_async_tmem_load()
        self.store_scores(values, coords, index, p, logical_page, scores, row, hi)

    @cute.jit
    def store_scores(self, values, coords, index, p, logical_page, scores, row, hi):
        """Publish this thread's token score of tile `p`, or -inf for a
        token that is not in the pool or lies past the row's end."""
        value = self.reduce_heads(values)
        m = coords[0][0]
        token = logical_page * 128 + m
        selected_block = index[0, p * 16 + m // 8]
        logical_token = selected_block * 8 + token % 8
        if selected_block < 0 or logical_token >= hi:
            value = Float32(-float("inf"))
        # `token` is below the pool width (see the class pre-condition).
        scores[row, token] = value

    @cute.jit
    def reduce_heads(self, values):
        """The mean over the query heads of the scaled, rectified scores."""
        sums = cute.make_rmem_tensor((4,), Float32)
        sums.fill(0)
        weight = Float32(self.head_dim**-0.5 / self.heads)
        for h in cutlass.range_constexpr(self.heads):
            x = values[h]
            sums[h % 4] = cute.math.fma(x + cute.math.absf(x), weight, sums[h % 4])
        return ((sums[0] + sums[2]) + (sums[1] + sums[3])) * Float32(0.5)

    @cute.jit
    def scale_tmem(self, pool, mma, sal, sbl):
        """Allocate the key and query scale factors in tensor memory."""
        sfat = pool.allocate_tensor(
            bs.make_tmem_layout_sfa(
                mma, self.tile, self.sf_vec, cute.slice_(sal, (None, None, None, 0))
            ),
            self.sf_dtype,
        )
        sfbt = pool.allocate_tensor(
            bs.make_tmem_layout_sfb(
                mma, self.tile, self.sf_vec, cute.slice_(sbl, (None, None, None, 0))
            ),
            self.sf_dtype,
        )
        return sfat, sfbt

    @cute.jit
    def scale_copy(self, source, destination):
        """The copy of scale factors from shared memory `source` to tensor
        memory `destination`, with its partitioned source and destination."""
        sfop = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE), self.sf_dtype
        )
        copy = tcgen05.make_s2t_copy(sfop, cute.filter_zeros(destination))
        thread = copy.get_slice(0)
        src = tcgen05.get_s2t_smem_desc_tensor(
            copy, thread.partition_S(cute.filter_zeros(source))
        )
        dst = thread.partition_D(cute.filter_zeros(destination))
        return copy, src, dst

    @cute.jit
    def load_query(self, tb, thr, sb, sfb, qs, row, tid, warp, bars, s2tb, sfbs, sfbd):
        """Load query `row` and its scale factors into shared memory and
        copy the scale factors on to tensor memory."""
        gb = cute.local_tile(
            tb.tma_tensor, (self.query_width, self.head_dim), (None, None, None)
        )
        sbs, gbg = cpasync.tma_partition(
            tb.atom,
            0,
            cute.make_layout(1),
            cute.group_modes(sb, 0, 3),
            cute.group_modes(thr.partition_B(gb), 0, 3),
        )
        sfbw = cute.make_tensor(
            cute.recast_ptr(sfb.iterator, dtype=Uint32),
            cute.make_layout((128 * self.sf_words,)),
        )
        for half in cutlass.range_constexpr(self.sf_words):
            sfbw[tid + half * 128] = Uint32(0)
        cute.arch.barrier()
        if tid < self.query_width:
            for half in cutlass.range_constexpr(self.sf_words):
                sfbw[half * 128 + tid * 4] = Uint32(qs[row, tid, half])
        if tid == 0:
            cute.arch.mbarrier_arrive_and_expect_tx(
                bars.iterator, cute.size_in_bytes(cutlass.Float4E2M1FN, tb.smem_layout)
            )
        if warp == 0:
            cute.copy(
                tb.atom,
                gbg[(None, 0, 0, row)],
                sbs[(None, 0)],
                tma_bar_ptr=bars.iterator,
            )
        cute.arch.mbarrier_wait(bars.iterator, 0)
        cute.arch.barrier()
        cute.arch.fence_view_async_shared()
        if warp == 0:
            cute.copy(s2tb, sfbs[(None, None, None, None, 0)], sfbd)
