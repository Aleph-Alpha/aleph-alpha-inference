# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""NVFP4 tensor-core scans of block summaries and of short contexts, built on
the scan pipeline of `SelectedPoolScan`."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import torch
from cutlass import Float32, Int32, Uint32
from cutlass.cute.nvgpu import cpasync, tcgen05

from aleph_alpha_inference.kernels.decode_reindex import SelectedPoolScan
from aleph_alpha_inference.kernels.summary_cache import SummaryCache

# Tokens per summary block. A 128-token page holds one 8-summary box.
BLOCK = SummaryCache.BLOCK


class BlockMeanScan(SelectedPoolScan):
    """Score the summary blocks of rows whose context exceeds the candidate
    pool, up to ceil(hi / BLOCK). Pool selection reads the scores of exactly
    these rows, so `pool_tokens` must be the served pool size: other rows
    are skipped and their scores are left unwritten."""

    def __init__(self, pool_tokens=16384):
        super().__init__(short_tokens=pool_tokens)
        self.block_size = BLOCK

    @cute.jit
    def stage_index(self, sm, k, table, selected, requests, lengths, row, chunk, tid):
        # Summary blocks need no index: every tile up to the row's end is live.
        return None, True, lengths[row]

    @cute.jit
    def tile_live(self, index, p):
        return True

    @cute.jit
    def locate(self, index, table, request, k, hi, local, logical, m):
        token = Uint32(logical * 128 + m)
        physical = Int32(0)
        slot = Uint32(0)
        if token < hi // self.block_size:
            physical = table[request, token // k.shape[1]]
            slot = token % k.shape[1]
        return physical, slot, token < hi // self.block_size

    @cute.jit
    def scan_length(self, hi, selected, scores):
        return cutlass.min(scores.shape[1], cute.ceil_div(hi, self.block_size))

    @cute.jit
    def store_scores(self, values, coords, index, p, logical_page, scores, row, hi):
        value = self.reduce_heads(values)
        block = logical_page * 128 + coords[0][0]
        if block >= hi // self.block_size:
            value = Float32(0)
        if block < scores.shape[1]:
            scores[row, block] = value


class GroupedScan(SelectedPoolScan):
    """Score `group` consecutive queries per CTA against shared keys (N = 16 * group).

    Prefill queries of one request read the same keys, so one key tile serves
    the whole group. Every output is the same block-scaled K=128 dot product
    with the same FP32 head reduction as the per-query scans, so scores match
    them bit for bit. Rows of different requests in a group run as separate
    segments. Warps 0-3 reduce scores, warps 4-7 load keys, warp 8 issues MMAs.

    Subclasses define the key tiles (`row_tiles`, `live_rows`, `load_tile`)
    and the published value of rows past the live keys (`dead`).
    """

    def __init__(self):
        super().__init__()
        # Queries per CTA.
        self.group = 4
        self.query_width = 16 * self.group
        self.tile = (128, self.query_width, 128)
        # Key tiles per unit of work, and key stages of the ring.
        self.tiles = 16
        self.stages = 8
        # Four loader warps follow the four reduction warps; the MMA warp is
        # last. Keys load as 512-byte TMA boxes.
        self.loaders = 4
        self.loader_warp = 4
        self.mma_warp = self.loader_warp + self.loaders
        self.block_threads = 32 * (self.mma_warp + 1)
        # Two accumulators let the MMA of a tile overlap the previous tile's
        # TMEM drain; occupancy is bound by registers and shared memory anyway.
        self.acc_stages = 2
        columns = self.acc_stages * self.query_width + 32
        self.tmem_columns = 1 << (columns - 1).bit_length()
        # Persistent CTAs (two per SM fit registers and shared memory) walk
        # contiguous runs of (query group, tile chunk) units, so a CTA sets up
        # barriers and TMEM once and reloads the query only per new group.
        device = torch.cuda.current_device()
        self.ctas = 2 * torch.cuda.get_device_properties(device).multi_processor_count

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
        ta = self.block_tma(k, al)
        units = cute.ceil_div(scores.shape[1], self.tiles * 128) * cute.ceil_div(
            scores.shape[0], self.group
        )
        self.grouped_kernel(
            mma, ta, tb, qs, k, ks, table, requests, lengths, scores, al, bl, sal, sbl
        ).launch(
            grid=(cutlass.min(units, Int32(self.ctas)), 1, 1),
            block=(self.block_threads, 1, 1),
            stream=stream,
        )

    @cute.jit
    def query_tensor(self, q):
        # Rows past the last query are out of bounds and zero-filled by TMA.
        return cute.make_tensor(
            cute.recast_ptr(q.iterator, dtype=cutlass.Float4E2M1FN),
            cute.make_layout((q.shape[0] * 16, 128, 1), stride=(128, 1, 0)),
        )

    @cute.jit
    def block_tma(self, k, al):
        """TMA over 8-row key blocks: (row, dim, block in page, page)."""
        fp4 = cutlass.Float4E2M1FN
        blocks = cute.make_tensor(
            cute.recast_ptr(k.iterator, dtype=fp4),
            cute.make_layout(
                (8, 128, k.shape[1] // 8, k.shape[0]),
                stride=(k.stride[1] * 2, 1, k.stride[1] * 16, k.stride[0] * 2),
            ),
        )
        box = cute.make_composed_layout(
            al.inner, 0, cute.make_layout((8, 128), stride=(128, 1))
        )
        return cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), blocks, box, (8, 128)
        )

    @cute.jit
    def load_rows(self, requests, lengths, scores, first):
        """Registers holding each row's request, end, tiles and segment index.

        Loading these once keeps dependent global loads out of every later
        segment query (they dominated the kernel before)."""
        request = cute.make_rmem_tensor((self.group,), Int32)
        hi = cute.make_rmem_tensor((self.group,), Int32)
        for i in cutlass.range_constexpr(self.group):
            request[i] = Int32(-1)
            hi[i] = Int32(0)
            if first + i < scores.shape[0]:
                request[i] = requests[first + i]
                hi[i] = lengths[first + i]
        tiles = cute.make_rmem_tensor((self.group,), Int32)
        segment = cute.make_rmem_tensor((self.group,), Int32)
        seg = Int32(0)
        for i in cutlass.range_constexpr(self.group):
            tiles[i] = Int32(0)
            if request[i] >= 0:
                tiles[i] = self.row_tiles(hi[i], scores)
            # A row's segment is the number of request changes before it.
            if cutlass.const_expr(i > 0):
                if request[i] != request[i - 1]:
                    seg += 1
            segment[i] = seg
            if request[i] < 0:
                segment[i] = Int32(-1)
        return request, hi, tiles, segment

    @cute.jit
    def segment(self, rows, s, chunk):
        """Rows [start, stop) of segment `s`, its request, live keys and tiles."""
        requests, hi, row_tiles, segments = rows
        request = Int32(0)
        start = Int32(-1)
        stop = Int32(-1)
        live = Int32(0)
        tiles = Int32(0)
        for i in cutlass.range_constexpr(self.group):
            if segments[i] == s:
                if start < 0:
                    start = Int32(i)
                    request = requests[i]
                stop = Int32(i + 1)
                live = cutlass.max(live, self.live_rows(hi[i]))
                tiles = cutlass.max(tiles, row_tiles[i])
        begin = chunk * self.tiles
        count = cutlass.max(Int32(0), cutlass.min(Int32(self.tiles), tiles - begin))
        if start < 0:
            count = Int32(0)
        return start, stop, request, live, count

    @cute.jit
    def tail(self, rows, scores, first, chunk, tid):
        """Publish rows' scores past their scanned tiles (nothing by default)."""
        pass

    @cute.kernel
    def grouped_kernel(  # noqa: C901, PLR0912, PLR0915 -- explicit asynchronous pipeline.
        self,
        mma: cute.TiledMma,
        ta: cpasync.TmaInfo,
        tb: cpasync.TmaInfo,
        qs: cute.Tensor,
        k: cute.Tensor,
        ks: cute.Tensor,
        table: cute.Tensor,
        requests: cute.Tensor,
        lengths: cute.Tensor,
        scores: cute.Tensor,
        al: cute.ComposedLayout,
        bl: cute.ComposedLayout,
        sal: cute.Layout,
        sbl: cute.Layout,
    ):
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        ctas, _, _ = cute.arch.grid_dim()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = tid % 32
        chunks = cute.ceil_div(scores.shape[1], self.tiles * 128)
        units = chunks * cute.ceil_div(scores.shape[0], self.group)
        per = cute.ceil_div(units, ctas)
        unit_begin = cta * per
        unit_end = cutlass.min(units, unit_begin + per)
        sm = cutlass.utils.SmemAllocator()
        # Rows of this CTA's groups: exit before any setup when none has a
        # tile to scan (e.g. no short rows at long contexts).
        flag = sm.allocate_tensor(Int32, cute.make_layout((1,)), byte_alignment=4)
        if tid == 0:
            flag[0] = Int32(0)
        cute.arch.barrier()
        first_row = unit_begin // chunks * self.group
        last_row = cutlass.min(
            (unit_end + chunks - 1) // chunks * self.group, scores.shape[0]
        )
        for row in cutlass.range(
            first_row + tid, last_row, self.block_threads, unroll=1
        ):
            if self.row_tiles(lengths[row], scores) > 0:
                flag[0] = Int32(1)
        cute.arch.barrier()
        if flag[0] != 0:
            self.scan_units(
                mma,
                ta,
                tb,
                qs,
                ks,
                table,
                requests,
                lengths,
                scores,
                al,
                bl,
                sal,
                sbl,
                sm,
                tid,
                warp,
                lane,
                chunks,
                unit_begin,
                unit_end,
            )

    @cute.jit
    def scan_units(  # noqa: C901, PLR0912, PLR0915 -- explicit asynchronous pipeline.
        self,
        mma,
        ta,
        tb,
        qs,
        ks,
        table,
        requests,
        lengths,
        scores,
        al,
        bl,
        sal,
        sbl,
        sm,
        tid,
        warp,
        lane,
        chunks,
        unit_begin,
        unit_end,
    ):
        # Page boxes rely on the SW64 pattern repeating every 512 bytes.
        sa = sm.allocate_tensor(cutlass.Float4E2M1FN, al.outer, 1024, swizzle=al.inner)
        sb = sm.allocate_tensor(cutlass.Float4E2M1FN, bl.outer, 128, swizzle=bl.inner)
        sfa = sm.allocate_tensor(self.sf_dtype, sal, 128)
        sfb = sm.allocate_tensor(self.sf_dtype, sbl, 128)
        num_bars = 1 + 2 * self.stages + 2 * self.acc_stages
        bars = sm.allocate_tensor(
            cutlass.Uint64, cute.make_layout((num_bars,)), byte_alignment=8
        )
        holder = sm.allocate_tensor(Int32, cute.make_layout((1,)), byte_alignment=4)
        pages = sm.allocate_tensor(
            Int32,
            cute.make_layout((self.loaders * self.tiles * 16,)),
            byte_alignment=16,
        )
        key_full = bars.iterator + 1
        key_empty = key_full + self.stages
        acc_full = key_empty + self.stages
        acc_empty = acc_full + self.acc_stages
        if tid == 0:
            cute.arch.mbarrier_init(bars.iterator, 1)
            for stage in cutlass.range_constexpr(self.stages):
                # Per loader warp: a cp.async arrival per lane plus one TMA
                # arrival.
                cute.arch.mbarrier_init(key_full + stage, 33 * self.loaders)
                cute.arch.mbarrier_init(key_empty + stage, 1)
            for slot in cutlass.range_constexpr(self.acc_stages):
                cute.arch.mbarrier_init(acc_full + slot, 1)
                cute.arch.mbarrier_init(acc_empty + slot, 4)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()
        allocator = cutlass.utils.TmemAllocator(
            holder.iterator,
            barrier_for_retrieve=pipeline.NamedBarrier(1, self.block_threads),
        )
        pool = allocator.reserve(self.tmem_columns)
        allocator.relinquish_alloc_permit()
        acc_layout = mma.make_fragment_C(
            mma.partition_shape_C((128, self.query_width))
        ).layout
        acc = pool.allocate_tensor(acc_layout, Float32)
        for _ in cutlass.range_constexpr(self.acc_stages - 1):
            pool.allocate_tensor(acc_layout, Float32)
        sfat, sfbt = self.scale_tmem(pool, mma, sal, sbl)
        thr = mma.get_slice(0)
        ar, br = mma.make_fragment_A(sa), mma.make_fragment_B(sb)
        gb = cute.local_tile(tb.tma_tensor, (self.query_width, 128), (None, None, None))
        sbs, gbg = cpasync.tma_partition(
            tb.atom,
            0,
            cute.make_layout(1),
            cute.group_modes(sb, 0, 3),
            cute.group_modes(thr.partition_B(gb), 0, 3),
        )
        s2ta, sfas, sfad = self.scale_copy(sfa, sfat)
        s2tb, sfbs, sfbd = self.scale_copy(sfb, sfbt)
        sfaw = cute.make_tensor(
            cute.recast_ptr(sfa.iterator, dtype=Uint32),
            cute.make_layout((self.stages * 128 * self.sf_words,)),
        )
        sfbw = cute.make_tensor(
            cute.recast_ptr(sfb.iterator, dtype=Uint32),
            cute.make_layout((128 * self.sf_words,)),
        )
        if warp < self.loader_warp:
            atom = cute.make_copy_atom(
                tcgen05.Ld32x32bOp(tcgen05.Repetition.x16, tcgen05.Pack.NONE),
                Float32,
            )
            tcopy = tcgen05.make_tmem_copy(atom, acc)
            tthr = tcopy.get_slice(tid)
            coords = tthr.partition_D(
                thr.partition_C(cute.make_identity_tensor((128, self.query_width)))
            )
            values = cute.make_rmem_tensor(coords.shape, Float32)
            p = Int32(0)
            for unit in cutlass.range(unit_begin, unit_end, unroll=1):
                chunk = unit % chunks
                first = unit // chunks * self.group
                rows = self.load_rows(requests, lengths, scores, first)
                self.tail(rows, scores, first, chunk, tid)
                _, row_hi, row_tiles, _ = rows
                for s in cutlass.range_constexpr(self.group):
                    start, stop, _, _, count = self.segment(rows, s, chunk)
                    tiles = cute.make_rmem_tensor((self.group,), Int32)
                    for i in cutlass.range_constexpr(self.group):
                        tiles[i] = row_tiles[i] if i >= start and i < stop else 0
                    for t in cutlass.range(count, unroll=1):
                        slot = p % self.acc_stages
                        cute.arch.mbarrier_wait(
                            acc_full + slot, (p // self.acc_stages) % 2
                        )
                        tile = cute.make_tensor(
                            acc.iterator + slot * self.query_width, acc.layout
                        )
                        cute.copy(tcopy, tthr.partition_S(tile), values)
                        cute.arch.fence_view_async_tmem_load()
                        if lane == 0:
                            cute.arch.mbarrier_arrive(acc_empty + slot)
                        logical = chunk * self.tiles + t
                        position = logical * 128 + coords[0][0]
                        for i in cutlass.range_constexpr(self.group):
                            if logical < tiles[i] and position < scores.shape[1]:
                                value = self.reduce_query(values, i * 16)
                                if position >= self.live_rows(row_hi[i]):
                                    value = self.dead()
                                scores[first + i, position] = value
                        p += 1
        elif warp < self.mma_warp:
            loader = warp - self.loader_warp
            # Each loader warp keeps its own copy of the staged pages.
            own = cute.make_tensor(
                pages.iterator + loader * self.tiles * 16,
                cute.make_layout((self.tiles * 16,)),
            )
            p = Int32(0)
            for unit in cutlass.range(unit_begin, unit_end, unroll=1):
                chunk = unit % chunks
                first = unit // chunks * self.group
                rows = self.load_rows(requests, lengths, scores, first)
                for s in cutlass.range_constexpr(self.group):
                    _, _, request, live, count = self.segment(rows, s, chunk)
                    # Earlier tiles must be issued before their pages change.
                    cute.arch.sync_warp()
                    self.resolve_pages(
                        own, table, request, live, count, chunk, loader, lane
                    )
                    cute.arch.sync_warp()
                    for t in cutlass.range(count, unroll=1):
                        stage, phase = p % self.stages, (p // self.stages) % 2
                        if p >= self.stages:
                            cute.arch.mbarrier_wait(key_empty + stage, phase ^ 1)
                        self.load_tile(
                            ta,
                            key_full + stage,
                            ks,
                            sa,
                            sfaw,
                            own,
                            t,
                            chunk * self.tiles + t,
                            live,
                            stage,
                            lane,
                            loader,
                        )
                        cute.arch.cp_async_mbarrier_arrive_noinc(key_full + stage)
                        p += 1
        elif warp == self.mma_warp:
            p = Int32(0)
            loaded = Int32(-1)
            queries = Int32(0)
            for unit in cutlass.range(unit_begin, unit_end, unroll=1):
                chunk = unit % chunks
                group = unit // chunks
                first = group * self.group
                rows = self.load_rows(requests, lengths, scores, first)
                total = Int32(0)
                for s in cutlass.range_constexpr(self.group):
                    _, _, _, _, count = self.segment(rows, s, chunk)
                    total += count
                if total > 0 and group != loaded:
                    if p > 0:
                        # Every MMA reading the previous query must finish.
                        last = p - 1
                        cute.arch.mbarrier_wait(
                            key_empty + last % self.stages, (last // self.stages) % 2
                        )
                    self.load_group_query(
                        tb, gbg, sbs, sfbw, qs, scores, bars, group, first, lane
                    )
                    cute.arch.mbarrier_wait(bars.iterator, queries % 2)
                    queries += 1
                    loaded = group
                    cute.arch.sync_warp()
                    cute.arch.fence_view_async_shared()
                    cute.copy(s2tb, sfbs[(None, None, None, None, 0)], sfbd)
                for _ in cutlass.range(total, unroll=1):
                    stage, key_phase = p % self.stages, (p // self.stages) % 2
                    cute.arch.mbarrier_wait(key_full + stage, key_phase)
                    slot = p % self.acc_stages
                    if p >= self.acc_stages:
                        cute.arch.mbarrier_wait(
                            acc_empty + slot, ((p // self.acc_stages) % 2) ^ 1
                        )
                    cute.arch.fence_view_async_shared()
                    cute.copy(s2ta, sfas[(None, None, None, None, stage)], sfad)
                    target = cute.make_tensor(
                        acc.iterator + slot * self.query_width, acc.layout
                    )
                    mma.set(tcgen05.Field.ACCUMULATE, False)
                    cute.gemm(
                        mma,
                        target,
                        [ar[(None, None, None, stage)], sfat],
                        [br[(None, None, None, 0)], sfbt],
                        target,
                    )
                    with cute.arch.elect_one():
                        tcgen05.commit(key_empty + stage)
                        tcgen05.commit(acc_full + slot)
                    p += 1
        cute.arch.barrier()
        allocator.free(pool.base_ptr)

    @cute.jit
    def load_group_query(
        self, tb, gbg, sbs, sfbw, qs, scores, bars, group, first, lane
    ):
        """Load a group's packed queries (TMA) and scales (by this warp)."""
        for i in cutlass.range_constexpr(128 * self.sf_words // 32):
            sfbw[i * 32 + lane] = Uint32(0)
        cute.arch.sync_warp()
        for i in cutlass.range_constexpr(cute.ceil_div(self.query_width, 32)):
            column = i * 32 + lane
            if column < self.query_width and first + column // 16 < scores.shape[0]:
                for half in cutlass.range_constexpr(self.sf_words):
                    word = Uint32(qs[first + column // 16, column % 16, half])
                    sfbw[half * 128 + (column % 32) * 4 + column // 32] = word
        if lane == 0:
            cute.arch.mbarrier_arrive_and_expect_tx(
                bars.iterator, cute.size_in_bytes(cutlass.Float4E2M1FN, tb.smem_layout)
            )
        cute.copy(
            tb.atom, gbg[(None, group, 0, 0)], sbs[(None, 0)], tma_bar_ptr=bars.iterator
        )

    @cute.jit
    def copy_box(self, ta, sa, stage, block):
        """TMA one 8-row block (512 bytes) into slot `block` of a stage tile."""
        gk = cute.group_modes(ta.tma_tensor, 0, 2)
        dst = cute.make_tensor(
            sa.iterator + cute.assume(stage * 16384 + block * 1024, divby=1024),
            cute.make_layout((8, 128), stride=(128, 1)),
        )
        ss, gs = cpasync.tma_partition(
            ta.atom, 0, cute.make_layout(1), cute.group_modes(dst, 0, 2), gk
        )
        return ss, gs

    @cute.jit
    def expect_boxes(self, full, tile, live, loader, lane):
        """Announce the bytes of this loader warp's live boxes of `tile` to the
        key barrier `full`."""
        count = Int32(0)
        for l in cutlass.range_constexpr(4):
            if tile * 128 + (loader * 4 + l) * 8 < live:
                count += 1
        if lane == 0:
            cute.arch.mbarrier_arrive_and_expect_tx(full, count * 512)

    @cute.jit
    def copy_scales(self, ks, sfaw, stage, m, page_physical, slot, valid_row):
        """Copy summary `m`'s scale words into stage `stage`, or zeros unless
        `valid_row`."""
        if not valid_row:
            page_physical = Int32(0)
            slot = Int32(0)
        self.load_key_scales(
            ks, sfaw, stage, (m % 32) * 4 + m // 32, page_physical, slot, valid_row
        )

    @cute.jit
    def reduce_query(self, values, offset: cutlass.Constexpr):
        # Packed form of SelectedPoolScan.reduce_heads: the same four RN FP32
        # chains over heads h % 4, then ((s0 + s2) + (s1 + s3)) * 0.5.
        sums = cute.make_rmem_tensor((4,), Float32)
        sums.fill(0)
        weight = Float32(self.head_dim**-0.5 / self.heads)
        for h in cutlass.range_constexpr(self.heads // 2):
            a, b = values[offset + h * 2], values[offset + h * 2 + 1]
            relu = cute.arch.add_packed_f32x2(
                (a, b), (cute.math.absf(a), cute.math.absf(b)), rnd="rn", ftz=False
            )
            sums[(h * 2) % 4], sums[(h * 2 + 1) % 4] = cute.arch.fma_packed_f32x2(
                relu,
                (weight, weight),
                (sums[(h * 2) % 4], sums[(h * 2 + 1) % 4]),
                rnd="rn",
                ftz=False,
            )
        x, y = cute.arch.add_packed_f32x2(
            (sums[0], sums[1]), (sums[2], sums[3]), rnd="rn", ftz=False
        )
        return (x + y) * Float32(0.5)


class GroupedBlockMeanScan(GroupedScan):
    """`BlockMeanScan` for prefill, scoring 128-summary tiles.

    A tile is 16 boxes of 8 summaries; a page holds one box. Loader warp w
    owns boxes 4w..4w+3 of every tile: one 512-byte TMA box each when live,
    plus the scales of summaries lane + 32 w (in them).
    Summaries past `live` in a live page hold stale data; they only feed their
    own scores, which are published as 0. Their scales are zero-filled.
    """

    def __init__(self, pool_tokens=16384):
        super().__init__()
        self.block_size = BLOCK
        self.pool_tokens = pool_tokens
        # Summaries per 128-token page.
        self.summaries = 128 // BLOCK

    @cute.jit
    def row_tiles(self, hi, scores):
        # As BlockMeanScan: only rows past the pool, up to ceil(hi / B).
        tiles = Int32(0)
        if hi > self.pool_tokens:
            tiles = cute.ceil_div(
                cutlass.min(scores.shape[1], cute.ceil_div(hi, self.block_size)), 128
            )
        return tiles

    @cute.jit
    def live_rows(self, hi):
        return hi // self.block_size

    @cute.jit
    def dead(self):
        return Float32(0)

    @cute.jit
    def resolve_pages(self, pages, table, request, live, count, chunk, loader, lane):
        # Resolve the pages of this warp's boxes up front: a table load per
        # tile would serialize its latency. A box is 8 summaries in one page.
        for i in cutlass.range_constexpr(cute.ceil_div(self.tiles * 4, 32)):
            index = i * 32 + lane
            t = index // 4
            j = loader * 4 + index % 4
            first_summary = (chunk * self.tiles + t) * 128 + j * 8
            physical = Int32(0)
            if t < count and first_summary < live:
                physical = table[request, first_summary // self.summaries]
            if index < self.tiles * 4:
                pages[t * 16 + j] = physical

    @cute.jit
    def load_tile(
        self, ta, full, ks, sa, sfaw, pages, t, tile, live, stage, lane, loader
    ):
        physical = pages[t * 16 + loader * 4 + lane % 4]
        self.expect_boxes(full, tile, live, loader, lane)
        for l in cutlass.range_constexpr(4):
            j = loader * 4 + l
            page_physical = cute.arch.shuffle_sync(physical, l)
            if tile * 128 + j * 8 < live:
                ss, gs = self.copy_box(ta, sa, stage, j)
                # The box is the page's only one.
                cute.copy(ta.atom, gs[(None, 0, page_physical)], ss, tma_bar_ptr=full)
        # Summary m = lane + 32 w lies in box m // 8 = 4 w + lane // 8.
        m = lane + loader * 32
        self.copy_scales(
            ks,
            sfaw,
            stage,
            m,
            cute.arch.shuffle_sync(physical, lane // 8),
            (tile * 128 + m) % self.summaries,
            tile * 128 + m < live,
        )


class GroupedShortScan(GroupedScan):
    """`SelectedPoolScan` for prefill rows whose context fits the pool.

    Such a row's source pool is the identity over its context (pool
    selection copies short pools), so pool position p scores token p. One
    tile is one 128-token page: loader warp w copies its blocks 4w..4w+3 by
    TMA and the scales of tokens lane + 32 w. Positions at or past the row's
    end, up to the pool width, are -inf, as in `SelectedPoolScan`.
    """

    def __init__(self, pool_tokens=16384):
        super().__init__()
        self.pool_tokens = pool_tokens

    @cute.jit
    def row_tiles(self, hi, scores):
        tiles = Int32(0)
        if hi <= self.pool_tokens:
            tiles = cute.ceil_div(cutlass.min(hi, scores.shape[1]), 128)
        return tiles

    @cute.jit
    def live_rows(self, hi):
        return hi

    @cute.jit
    def dead(self):
        return Float32(-float("inf"))

    @cute.jit
    def tail(self, rows, scores, first, chunk, tid):
        # Tiles past a short row's end hold no tokens: publish -inf there.
        requests, hi, row_tiles, _ = rows
        for i in cutlass.range_constexpr(self.group):
            if requests[i] >= 0 and hi[i] <= self.pool_tokens:
                for t in cutlass.range(
                    cutlass.max(row_tiles[i], chunk * self.tiles),
                    cutlass.min(
                        Int32(self.tiles) * (chunk + 1), scores.shape[1] // 128
                    ),
                    unroll=1,
                ):
                    # Run by the 128 reduction threads.
                    scores[first + i, t * 128 + tid] = Float32(-float("inf"))

    @cute.jit
    def resolve_pages(self, pages, table, request, live, count, chunk, loader, lane):
        # Each loader warp keeps its own copy of the segment's pages.
        for i in cutlass.range_constexpr(cute.ceil_div(self.tiles, 32)):
            t = i * 32 + lane
            page = chunk * self.tiles + t
            physical = Int32(0)
            if t < count and page * 128 < live:
                physical = table[request, page]
            if t < self.tiles:
                pages[t] = physical

    @cute.jit
    def load_tile(
        self, ta, full, ks, sa, sfaw, pages, t, tile, live, stage, lane, loader
    ):
        physical = pages[t]
        self.expect_boxes(full, tile, live, loader, lane)
        for l in cutlass.range_constexpr(4):
            j = loader * 4 + l
            if tile * 128 + j * 8 < live:
                ss, gs = self.copy_box(ta, sa, stage, j)
                cute.copy(ta.atom, gs[(None, j, physical)], ss, tma_bar_ptr=full)
        m = lane + loader * 32
        self.copy_scales(ks, sfaw, stage, m, physical, m, tile * 128 + m < live)
