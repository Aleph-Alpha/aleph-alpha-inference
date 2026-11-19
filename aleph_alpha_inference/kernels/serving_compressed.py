# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""Sparse attention of the served model over a paged NVFP4 cache.

Per query row, a source layer scores the block summaries of the row's context
and expands the best blocks to a token pool. Every layer scores the pool
tokens exactly, selects the best ones and attends over them, the local window
and the sink tokens.
"""

from functools import lru_cache

import torch

from aleph_alpha_inference.kernels.arch import is_blackwell
from aleph_alpha_inference.kernels.cute_jit import Executables
from aleph_alpha_inference.kernels.decode_pack import DecodeQueryPack
from aleph_alpha_inference.kernels.decode_reindex import SelectedPoolScan
from aleph_alpha_inference.kernels.decode_selection import (
    ExpandedPoolSelection,
    RadixSelection,
)
from aleph_alpha_inference.kernels.nvfp4_attention import (
    native as nvfp4_native,
    nvfp4_attention,
)
from aleph_alpha_inference.kernels.prefill_staging import stage_context
from aleph_alpha_inference.kernels.routing_scan_mma import MmaPoolScan, MmaSummaryScan
from aleph_alpha_inference.kernels.sparse_decode import SparseDecodeAttention
from aleph_alpha_inference.kernels.summary_cache import SummaryCache
from aleph_alpha_inference.kernels.summary_scan import (
    BlockMeanScan,
    GroupedBlockMeanScan,
    GroupedShortScan,
)
from aleph_alpha_inference.kernels.union_attention import (
    native as union_native,
    TILE_ROWS,
    union_attention,
    union_tiles,
    union_workspace,
)

# Longest prefill context served by union tiles (validated in serving).
SERVING_UNION_SPAN = 1 << 19
# Tokens per summary block.
BLOCK = SummaryCache.BLOCK
# Tokens per entry of a source pool.
POOL_GROUP = 8
# Besides its selected tokens every row attends the last LOCAL_TOKENS tokens
# of its context and the first SINK_TOKENS ones.
LOCAL_TOKENS = 128
SINK_TOKENS = 8
# Rows routed per kernel launch. This bounds the score scratch.
CHUNK_ROWS = 1024
# Decode selects blocks per segment of SEGMENT_BLOCKS once a context has more
# than SEGMENTED_ABOVE blocks.
SEGMENT_BLOCKS = 8192
SEGMENTED_ABOVE = 32768
_COMPILED = Executables("compressed_routing")


def route_width(topk):
    """Number of tokens a row attends when it selects `topk` tokens."""
    return topk + LOCAL_TOKENS + SINK_TOKENS


def score_blocks(span, decode):
    """Columns of the block score scratch for contexts of up to `span` tokens.

    Decode pads contexts past `SEGMENTED_ABOVE` blocks to whole segments, so
    that every context limit selects per segment.
    """
    width = -(-span // (BLOCK * 128)) * 128
    if decode and width > SEGMENTED_ABOVE:
        width = -(-width // SEGMENT_BLOCKS) * SEGMENT_BLOCKS
    return width


def row_bounds(cu, seq_lens, requests, lengths):
    """Find the request and the causal context length of every query row.

    Args:
        cu: First query row of each request and one end entry,
            non-decreasing (vLLM's `query_start_loc`).
        seq_lens: Context length of each request after this step.
        requests: Output, the request id per row.
        lengths: Output, the number of context tokens each row may attend.
            A request's last row sees its whole context, each earlier row
            one token less.

    Rows at or past `cu[-1]` are padding and get request 0 and length 0.
    """
    rows = torch.arange(requests.shape[0], device=cu.device, dtype=cu.dtype)
    # Zero for padding rows.
    live = rows < cu[-1]
    # The last request that starts at or before the row; this skips empty
    # requests, which share their start with the next one.
    request = torch.searchsorted(cu, rows, right=True, out_int32=True) - 1
    torch.mul(request, live, out=requests)
    length = seq_lens[requests] - (cu[requests + 1] - rows - 1)
    torch.mul(length, live, out=lengths)


@lru_cache(None)
def routing_workspace(device, owner, rows, span, topk, decode, pool_tokens=16384):
    """Bounded scratch shared by sequential layers in one vLLM workspace lane.

    Prefill and decode each size one arena for their maximum row count. Runtimes
    take row views, so new batch sizes never allocate device memory.

    Source pools have a separate lifetime: consumers may still need them after
    another layer has used this scratch. Keep those pools outside this arena.

    Args:
        device: Device of the scratch.
        owner: Identifies the workspace lane; each owner gets its own arena.
        rows: Largest number of query rows per step.
        span: Longest context in tokens.
        topk: Tokens selected per row.
        decode: Whether the arena serves decode steps or prefill steps.
        pool_tokens: Tokens in the pool that a source layer selects.

    Returns:
        The scratch tensors by name.
    """

    def empty(*shape, dtype=torch.int32):
        return torch.empty(*shape, device=device, dtype=dtype)

    chunk = min(rows, CHUNK_ROWS)
    return dict(
        qp=torch.zeros(rows, 16, 64, device=device, dtype=torch.int8),
        qs=torch.zeros(rows, 16, 2, device=device, dtype=torch.int32),
        requests=empty(rows),
        hi=empty(rows),
        small=empty(rows, pool_tokens // BLOCK),
        remote=empty(rows, topk),
        route=empty(rows, route_width(topk)),
        used=empty(rows),
        block_scores=empty(chunk * score_blocks(span, decode), dtype=torch.float32),
        token_scores=empty(chunk * pool_tokens, dtype=torch.float32),
        # Only prefill attends over union tiles.
        union=None if decode else union_workspace(rows, device),
    )


@lru_cache(None)
def source_pool(device, owner, source, rows, pool_tokens=16384):
    """The token pool that the source layer `source` selects for up to `rows` rows.

    Each `(owner, source)` pair gets its own tensor. Its consumers read it
    after other layers have reused the routing scratch.
    """
    return torch.empty(
        rows, pool_tokens // POOL_GROUP, device=device, dtype=torch.int32
    )


def routing_scans(decode, pool_tokens):
    """The summary scan, the short-row scan and the pool refinement scan that
    this GPU runs, and the launch key of the refinement scan."""
    if is_blackwell():
        # Prefill rows of one request share summaries: score query groups.
        # Rows past the pool are scored; pool selection reads only those.
        mean = (
            BlockMeanScan(pool_tokens) if decode else GroupedBlockMeanScan(pool_tokens)
        )
        # Prefill rows whose context fits the pool have identity pools: one
        # grouped dense scan scores them, the pool scan the rest.
        short = None if decode else GroupedShortScan(pool_tokens)
        refine = SelectedPoolScan(
            short_tokens=0 if decode else pool_tokens,
            pages=8 if decode else 16,
        )
        return mean, short, refine, (refine.short_tokens, refine.pages)
    # Without FP4 tensor cores the per-row mma.sync scans serve decode and
    # prefill rows alike, every row from its (identity) pool.
    refine = MmaPoolScan()
    return MmaSummaryScan(pool_tokens), None, refine, refine.key


class ServingCompressedLayer:
    """Routing and attention of one sparse layer for a fixed number of rows.

    The layer owns no device memory. It views `scratch`, which it shares with
    the other layers of its workspace lane.

    Args:
        cache: The `SummaryCache` that holds the context.
        table: Pages of each request, [requests, pages] int32.
        rows: Number of query rows per step.
        span: Longest context in tokens.
        topk: Tokens selected per row.
        decode: Whether every request has one query row. Decode and prefill
            use different kernels.
        scratch: A `routing_workspace` of at least `rows` rows and `span`
            tokens, with the same `topk`, `decode` and `pool_tokens`.
        source_pool: The `source_pool` of a layer that runs as source, with
            the same `pool_tokens`.
        allow_staging: Whether prefill may stage contexts to attend over
            union tiles.
        pool_tokens: Tokens in the pool that a source layer selects.

    Raises:
        ValueError: If `scratch` or `source_pool` is too small or was sized
            for other settings.
    """

    def __init__(
        self,
        cache,
        table,
        rows,
        span,
        topk,
        decode,
        *,
        scratch,
        source_pool=None,
        allow_staging=True,
        pool_tokens=16384,
    ):
        self.pool_tokens = pool_tokens
        self.cache, self.table, self.rows = cache, table, rows
        self.span, self.topk, self.decode = span, topk, decode
        self.chunk = min(rows, CHUNK_ROWS)
        width = score_blocks(span, decode)
        topk_blocks = pool_tokens // BLOCK
        if (
            rows > scratch["qp"].shape[0]
            or self.chunk * width > scratch["block_scores"].numel()
            or self.chunk * pool_tokens > scratch["token_scores"].numel()
            or scratch["remote"].shape[1] != topk
            or scratch["small"].shape[1] != topk_blocks
            or (
                source_pool is not None
                and (
                    rows > source_pool.shape[0]
                    or source_pool.shape[1] != pool_tokens // POOL_GROUP
                )
            )
        ):
            raise ValueError(
                "Compressed routing scratch capacity exceeded or sized for "
                "other settings"
            )
        for name in (
            "qp",
            "qs",
            "requests",
            "hi",
            "small",
            "remote",
            "route",
            "used",
        ):
            setattr(self, name, scratch[name][:rows])
        self.pool = None if source_pool is None else source_pool[:rows]
        self.block_scores = scratch["block_scores"][: self.chunk * width].view(
            self.chunk, width
        )
        self.token_scores = scratch["token_scores"][: self.chunk * pool_tokens].view(
            self.chunk, pool_tokens
        )
        # Prefill tiles of >= TILE_ROWS queries of one request attend over the
        # union of their keys (FA4 on the staged BF16 context), up to the
        # span validated in serving.
        self.union = None
        if not decode:
            nvfp4_native()  # Load (or build) before serving, not mid-step.
            if allow_staging and span <= SERVING_UNION_SPAN:
                union_native()
                self.union = scratch["union"]
        self.pack = DecodeQueryPack()
        self.mean, self.short, self.refine, self.refine_key = routing_scans(
            decode, pool_tokens
        )
        self.pool_select = ExpandedPoolSelection(topk_blocks, BLOCK)
        # Past SEGMENTED_ABOVE summary blocks (512k-token spans) one CTA per
        # decode row can no longer hold its keys in registers and rereads them
        # per radix pass, with too few rows to fill the GPU. Select each
        # segment's top blocks instead, then merge the candidates (exact,
        # with index-ordered ties). The token score scratch holds the
        # candidates; refinement overwrites it afterwards.
        self.segments = (
            width // SEGMENT_BLOCKS if decode and width > SEGMENTED_ABOVE else 1
        )
        if self.segments * topk_blocks > pool_tokens:
            self.segments = 1
        if self.segments > 1:
            self.segment_select = RadixSelection(
                0,
                threads=512,
                segments=self.segments,
                topk=topk_blocks,
                block_size=BLOCK,
            )
            self.merge_select = ExpandedPoolSelection(topk_blocks, BLOCK, merge=True)
        self.token_select = RadixSelection(
            2, threads=512, topk=topk, local=LOCAL_TOKENS, sinks=SINK_TOKENS
        )
        self.attention = (
            SparseDecodeAttention(rows, cache, total=route_width(topk))
            if decode
            else None
        )

    def _select_pool(self, sl, n, hi):
        """Select summary blocks and write the expanded token pool of rows `sl`."""
        target = self.small[sl]
        # Pool selection writes the expanded token pool itself.
        if self.segments > 1:
            s_count, width = self.segments, self.block_scores.shape[1]
            k_blocks = self.pool_tokens // BLOCK
            flat = self.token_scores.view(-1)[: n * s_count * k_blocks]
            candidates = flat.view(torch.int32).view(n, s_count * k_blocks)
            segment_out = candidates.view(n * s_count, k_blocks)
            segment_scores = self.block_scores[:n].view(n * s_count, width // s_count)
            _COMPILED.launch(
                self.segment_select,
                (segment_scores, segment_out, hi, segment_out, segment_out, hi),
                key=(k_blocks, s_count),
            )
            _COMPILED.launch(
                self.merge_select,
                (self.block_scores[:n], candidates, hi, target, self.pool[sl], hi),
                key=(k_blocks, "merge"),
            )
        else:
            _COMPILED.launch(
                self.pool_select,
                (self.block_scores[:n], target, hi, target, self.pool[sl], hi),
                key=(self.pool_tokens // BLOCK,),
            )

    def _score_pool(self, sl, n, qp, qs, req, hi):
        """Score the pool tokens of rows `sl` into the token scratch."""
        tensors = (
            qp,
            qs,
            self.cache.rk.view(torch.uint8),
            self.cache.rks,
            self.table,
            req,
            self.pool[sl],
            hi,
            self.token_scores[:n],
        )
        # The short scan writes no row whose context exceeds the pool, so it
        # has nothing to do when every row's context does.
        if self.short is not None and self.min_context <= self.pool_tokens:
            _COMPILED.launch(self.short, tensors, key=(self.pool_tokens,))
        # The span bounds every row's context: when it fits the pool, the
        # short scan already scored every row.
        if self.short is None or self.span > self.pool_tokens:
            _COMPILED.launch(self.refine, tensors, key=self.refine_key)

    def run(
        self,
        q,
        cu,
        ends,
        *,
        role,
        out,
        bounds,
        query_starts,
        pool=None,
        attention_workspace=None,
        min_context=0,
    ):
        """Route every query row and attend over its selected tokens.

        Args:
            q: Queries, [rows, 1, 48, 128] BF16.
            cu: First query row of each request and one end entry, int32
                (vLLM's `query_start_loc`).
            ends: Context length of each request after this step, int32.
            role: "source" selects the token pool from the block summaries;
                "reindex" reuses the pool of its source layer.
            out: Receives the attention output, shaped like `q`.
            bounds: A list shared by the layers of one step. The first layer
                to run fills it with the rows' request ids and causal lengths
                (see `row_bounds`) and the others reuse them.
            query_starts: `cu` as a tuple of host integers. Prefill uses it to
                find the requests that attend over union tiles; decode ignores
                it.
            pool: The source layer's pool, for the "reindex" role.
            attention_workspace: Decode attention scratch (see
                `sparse_decode.workspace_shapes`).
            min_context: A lower bound on the context length of every query
                row, or 0 if unknown. Prefill skips scoring work that no row
                of at least this context needs.

        Returns:
            `out`.
        """
        self.select(
            q, cu, ends, role=role, bounds=bounds, pool=pool, min_context=min_context
        )
        if self.decode:
            self.attention(
                q,
                self.table,
                self.route,
                self.used,
                self.requests,
                out=out,
                workspace=attention_workspace,
            )
        else:
            self._attend_prefill(q, ends, query_starts, out)
        return out

    def select(self, q, cu, ends, *, role, bounds, pool=None, min_context=0):
        """Write each row's selected tokens to `route` and `used`, without
        attending. A source layer also writes its pool. See `run` for the
        arguments."""
        if not bounds:
            row_bounds(cu, ends, self.requests, self.hi)
            bounds.extend((self.requests, self.hi))
        self.requests, self.hi = bounds
        # `bounds` is one object per step; it keys the staged context.
        self.step, self.min_context = bounds, min_context
        self.pack(q, self.qp, self.qs)
        if pool is not None:
            self.pool = pool
        for start in range(0, self.rows, self.chunk):
            stop = min(start + self.chunk, self.rows)
            sl = slice(start, stop)
            n = stop - start
            qp, qs, req, hi = (
                self.qp[sl].view(torch.uint8),
                self.qs[sl],
                self.requests[sl],
                self.hi[sl],
            )
            if role == "source":
                # The span bounds every row's context. Pool selection reads
                # block scores only of rows past the pool, so the block-mean
                # scan has nothing to score unless the span exceeds it.
                # Selection still runs: it writes the other rows' identity
                # pools.
                if self.span > self.pool_tokens:
                    _COMPILED.launch(
                        self.mean,
                        (
                            qp,
                            qs,
                            self.cache.summaries,
                            self.cache.summary_scales,
                            self.table,
                            req,
                            self.small[sl],
                            hi,
                            self.block_scores[:n],
                        ),
                        key=(self.pool_tokens,),
                    )
                self._select_pool(sl, n, hi)
            self._score_pool(sl, n, qp, qs, req, hi)
            _COMPILED.launch(
                self.token_select,
                (
                    self.token_scores[:n],
                    self.pool[sl],
                    hi,
                    self.remote[sl],
                    self.route[sl],
                    self.used[sl],
                ),
                key=(self.topk,),
            )

    def _attend_rows(self, q, start, end, out, **kwargs):
        """Attend rows `start` to `end` with the per-row NVFP4 kernel."""
        if start < end:
            sl = slice(start, end)
            nvfp4_attention(
                q[sl],
                self.cache,
                self.table,
                self.requests[sl],
                self.route[sl],
                self.used[sl],
                out=out[sl],
                **kwargs,
            )

    def _attend_prefill(self, q, ends, query_starts, out):
        """Union tiles for requests with >= TILE_ROWS queries; the direct NVFP4
        kernel for the other rows and for tiles whose union is too large."""
        if self.union is None:
            self._attend_rows(q, 0, self.rows, out)
            return
        begin = 0
        for request, (start, end) in enumerate(zip(query_starts, query_starts[1:])):
            if end - start < TILE_ROWS:
                continue
            self._attend_rows(q, begin, start, out)
            sl = slice(start, end)
            k, v = stage_context(
                self.cache, self.table, ends, request, self.span, step=self.step
            )
            tiles = union_tiles(self.route[sl], self.used[sl], self.span, self.union)
            union_attention(q[sl, 0], k, v, tiles, out=out[sl, 0])
            self._attend_rows(
                q, start, end, out, only_tiles=tiles[2], tile_rows=TILE_ROWS
            )
            begin = end
        self._attend_rows(q, begin, self.rows, out)
