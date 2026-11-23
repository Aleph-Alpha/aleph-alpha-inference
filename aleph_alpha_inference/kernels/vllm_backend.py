# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""Native vLLM paging for full-head FP4 routing and NVFP4 Kolibri KV storage."""

from collections import OrderedDict
from dataclasses import dataclass, field, replace

import torch
from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.utils.math_utils import round_up
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionImpl,
    AttentionMetadataBuilder,
)
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
from vllm.v1.kv_cache_interface import KVCacheLayout, KVQuantMode
from vllm.v1.worker import ubatching, workspace
from vllm.v1.worker.workspace import current_workspace_manager

from aleph_alpha_inference.kernels.sparse_decode import split_capacity, workspace_shapes
from aleph_alpha_inference.kernels.sparse_layout import SparseCheckpointLayout
from aleph_alpha_inference.kernels.summary_cache import SummaryCache


# Alignment in bytes of every cache field of every page.
PAGE_ALIGNMENT = 512


def record_fields(cache_type, page):
    """(start, width) of each `cache_type` constructor tensor within one page.

    A KV owner's page is one record: K/V, routing keys, block-mean summaries
    and the partial sum. Each field starts on a 512-byte
    boundary, so every field of every page is aligned when the block stride is.
    """
    fields, start = [], 0
    for width in cache_type.page_bytes(page):
        fields.append((start, width))
        start += round_up(width, PAGE_ALIGNMENT)
    return fields


def record_bytes(cache_type, page):
    """Bytes of one page record of `cache_type` (see `record_fields`)."""
    start, width = record_fields(cache_type, page)[-1]
    return start + round_up(width, PAGE_ALIGNMENT)


def _workspace_owner():
    # Match the installed vLLM workspace manager's (ubatch, lane) ownership.
    # Allocation addresses can change on growth, so they cannot identify owners.
    return ubatching.dbo_current_ubatch_id(), workspace._workspace_lane.get()


def min_row_context(query_start_loc, seq_lens):
    """The shortest context that any query row of a step attends.

    Args:
        query_start_loc: First query row of each request and one end entry,
            a host tensor.
        seq_lens: Context length of each request after the step, a host
            tensor, or None if unknown.

    Returns:
        The least context length over the first query rows of all requests
        that have query rows, or 0 if `seq_lens` is None or no request has
        query rows. Requests without query rows (CUDA-graph padding) do not
        count.
    """
    if seq_lens is None:
        return 0
    query_lens = query_start_loc.diff()
    live = query_lens > 0
    if not live.any():
        return 0
    # A request's first query row sees all but its later query rows.
    return int((seq_lens[live] - query_lens[live] + 1).min())


@dataclass
class Block8Metadata:
    """What a sparse layer needs of one step's batch.

    `max_seq_len` is the model's context limit for decode steps (every request
    has one query row) and the longest context of the step otherwise.
    `query_starts` holds `query_start_loc` as host integers for prefill steps.
    `min_context` is a lower bound on the context of every query row of a
    prefill step, or 0 if unknown.
    """

    num_actual_tokens: int
    max_query_len: int
    max_seq_len: int
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    query_starts: tuple[int, ...] = ()
    min_context: int = 0
    # Request id and causal length per query row, filled by the first sparse
    # layer of the step and shared by the others. `replace` keeps the list, so
    # the per-cache-group copies from `update_block_table` share it too.
    row_bounds: list[torch.Tensor] = field(default_factory=list)


class Block8MetadataBuilder(AttentionMetadataBuilder):
    """Builds `Block8Metadata` from vLLM's common attention metadata."""

    # Prefill plans its kernels on the host from the batch's query starts and
    # longest sequence. A FULL CUDA graph would replay the capture batch's plan
    # for every later prefill. Decode does not depend on either.
    _cudagraph_support = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    supports_update_block_table = True

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.max_model_len = vllm_config.model_config.max_model_len
        # Under async speculative decoding, `seq_lens_cpu_upper_bound`
        # overestimates decode rows. A lower bound taken from it could skip
        # rows that need scoring, so it is only trusted without speculation.
        self.exact_host_seq_lens = vllm_config.speculative_config is None

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        common = common_attn_metadata
        prefill = common.max_query_len > 1
        min_context = (
            min_row_context(common.query_start_loc_cpu, common.seq_lens_cpu_upper_bound)
            if prefill and self.exact_host_seq_lens
            else 0
        )
        return Block8Metadata(
            common.num_actual_tokens,
            common.max_query_len,
            self.max_model_len if common.max_query_len == 1 else common.max_seq_len,
            common.query_start_loc,
            common.seq_lens,
            common.block_table_tensor,
            common.slot_mapping,
            tuple(common.query_start_loc_cpu.tolist()) if prefill else (),
            min_context,
        )

    def update_block_table(self, metadata, blk_table, slot_mapping):
        return replace(metadata, block_table=blk_table, slot_mapping=slot_mapping)


# Layers select this class through `attn_backend=`, so it is not registered in
# vLLM's single global CUSTOM backend slot. `get_name` still returns "CUSTOM"
# because vLLM's Attention maps the name to an AttentionBackendEnum member.
class Block8Backend(FlashAttentionBackend):
    """Attention backend of the sparse layers: one `SummaryCache` record per
    KV page."""

    cache_type = SummaryCache

    @staticmethod
    def get_name():
        return "CUSTOM"

    @staticmethod
    def get_impl_cls():
        return Block8Impl

    @staticmethod
    def get_builder_cls():
        return Block8MetadataBuilder

    @classmethod
    def get_supported_kernel_block_sizes(cls, kv_cache_spec=None):
        return [128]

    @classmethod
    def supported_kv_cache_layouts(cls):
        # Only block-outermost layouts let the allocator pack SWA and sparse
        # pages without padding every SWA page to the sparse page size. With
        # one head slot per page, both are the same bytes.
        return (KVCacheLayout.BLHNC, KVCacheLayout.BLNHC)

    @classmethod
    def customize_spec(cls, spec):
        # One record per page holds all of the owner's cache fields (see
        # `record_fields`). Separate vLLM cache layers per field would make
        # vLLM's packed planner treat the owners' four page sizes as a 4-repeat
        # pattern and pack SWA layers in fours: a 512 KiB block stride for the
        # owners' 330 KiB of pages, a third of every long-context block unused.
        # With one page size per owner, SWA layers pack in pairs below it.
        page = record_bytes(cls.cache_type, spec.block_size)
        if page % spec.block_size:
            raise ValueError(f"{cls.cache_type.__name__} pages hold whole tokens")
        return replace(
            spec,
            dtype=torch.uint8,
            num_head_slots=1,
            state_content_bytes=page // spec.block_size,
            kv_quant_mode=KVQuantMode.NONE,
            # The kernels need aligned cache fields (checked in `_cache_view`).
            block_stride_alignment=PAGE_ALIGNMENT,
        )


def serving_backend(index_compression):
    """The attention backend class that serves sparse layers.

    Args:
        index_compression: Tokens per routing summary block, from the
            sparse layout of the HF config.

    Raises:
        ValueError: Unless `index_compression` is 16 (mean16 routing), the
            only policy the plugin serves.
    """
    if index_compression != 16:
        raise ValueError(
            "Kolibri sparse serving supports only mean16 routing "
            f"(index_compression=16); got {index_compression}"
        )
    return Block8Backend


class Block8Impl(AttentionImpl):
    """Sparse attention of one layer over its KV owner's `SummaryCache`.

    An indexer layer selects a token pool per query row; a reindexer layer
    reuses the pool of its `source_attention`.
    """

    supports_quant_query_input = False
    supports_dcp = False

    def __init__(
        self,
        num_heads,
        head_size,
        scale,
        num_kv_heads=None,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
        attn_type="decoder",
        kv_sharing_target_layer_name=None,
        **kwargs,
    ):
        if (num_heads, num_kv_heads, head_size) != (48, 4, 128):
            raise ValueError(
                "Block8 serving currently requires local Q48/KV4/D128 (TP1)"
            )
        if alibi_slopes is not None or sliding_window is not None or logits_soft_cap:
            raise ValueError(
                "Block8 replaces full attention without ALiBi or soft caps"
            )
        self.num_heads, self.num_kv_heads, self.head_size = (
            num_heads,
            num_kv_heads,
            head_size,
        )
        if kv_sharing_target_layer_name is not None:
            raise ValueError(
                "Kolibri KV-sharing layers read their source's cache "
                "(aleph_alpha_inference.model.vllm_shared_kv.SharedKVAttention)"
            )
        self.scale, self.kv_cache_dtype = scale, kv_cache_dtype
        self._cache = None
        self._decode_runtimes = OrderedDict()
        self._prefill_runtimes = OrderedDict()
        config = get_current_vllm_config()
        layout = SparseCheckpointLayout.from_hf_config(
            config.model_config.hf_text_config
        )
        # Tokens each row selects, and tokens of the pool it selects them from.
        self._topk = layout.index_topk
        self._pool_tokens = layout.index_pool_size
        self._allow_prefill_staging = not config.parallel_config.use_ubatching
        context_limit = config.model_config.max_model_len
        self._decode_span_limit = context_limit
        self._prefill_query_limit = config.scheduler_config.max_num_batched_tokens
        self._prefill_span_limit = max(
            16384, 1 << (max(1, context_limit) - 1).bit_length()
        )
        self._max_decode_rows = config.scheduler_config.max_num_seqs
        # Decode graphs can be captured beyond max_num_seqs; the shared decode
        # routing arena must cover every captured or eager decode batch.
        self._max_decode_batch = max(
            self._max_decode_rows,
            config.compilation_config.max_cudagraph_capture_size or 0,
        )
        self._routing_pools = {}

    @property
    def routing_pool(self):
        """The token pool this layer selected last in the current workspace lane."""
        return self._routing_pools.get(_workspace_owner())

    @routing_pool.setter
    def routing_pool(self, pool):
        self._routing_pools[_workspace_owner()] = pool

    def _cache_view(self, owner):
        """The cache of KV owner `owner`, from its layer's live binding."""
        kv_cache = get_forward_context().no_compile_layers[owner].kv_cache
        if self._cache is None or self._cache.storage.data_ptr() != kv_cache.data_ptr():
            if (
                kv_cache.data_ptr() % PAGE_ALIGNMENT
                or kv_cache.stride(0) % PAGE_ALIGNMENT
            ):
                raise ValueError(f"Block8 pages must be {PAGE_ALIGNMENT}-byte aligned")
            page = kv_cache.size(2)
            record = kv_cache.view(kv_cache.size(0), -1)
            if record.size(1) < record_bytes(SummaryCache, page):
                raise ValueError("Block8 pages must hold every cache field")
            self._cache = SummaryCache(
                *(
                    record[:, start : start + width]
                    for start, width in record_fields(SummaryCache, page)
                ),
                page=page,
            )
        return self._cache

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        """Append the step's keys and values to the layer's cache."""
        cache = self._cache_view(layer.layer_name)
        count = slot_mapping.numel()
        cache.append(key[:count], value[:count], slot_mapping)

    def _compressed_routing_workspace(self, device, decode):
        """The routing scratch of the current workspace lane and, for an
        indexer layer, its source pool."""
        from aleph_alpha_inference.kernels.serving_compressed import (
            routing_workspace,
            source_pool,
        )

        owner = _workspace_owner()
        rows, span = (
            (self._max_decode_batch, self._decode_span_limit)
            if decode
            else (self._prefill_query_limit, self._prefill_span_limit)
        )
        scratch = routing_workspace(
            device,
            owner,
            rows,
            span,
            self._topk,
            decode,
            self._pool_tokens,
        )
        pool = (
            source_pool(
                device,
                owner,
                (id(self), decode),
                rows,
                self._pool_tokens,
            )
            if self.sparse_role == "indexer"
            else None
        )
        return scratch, pool

    def _reserve_scratch(self, device):
        """Allocate the layer's scratch and build its native extensions, so
        that vLLM's memory profiling run accounts for them."""
        from aleph_alpha_inference.kernels.nvfp4_attention import native as nvfp4_native
        from aleph_alpha_inference.kernels.prefill_staging import workspace

        self._compressed_routing_workspace(device, decode=False)
        self._compressed_routing_workspace(device, decode=True)
        self._compressed_decode_workspace()
        nvfp4_native()
        if self._allow_prefill_staging:
            # Union-tile prefill attends over the staged BF16 context.
            workspace(device)
            from aleph_alpha_inference.kernels.union_attention import warm_up

            if torch.device(device).type == "cuda":
                warm_up(torch.device(device))

    def _compressed_decode_workspace(self):
        """The decode attention scratch of the current workspace lane."""
        from aleph_alpha_inference.kernels.serving_compressed import route_width

        splits = split_capacity(route_width(self._topk))
        return current_workspace_manager().get_simultaneous(
            *(
                (shape, torch.float32)
                for shape in workspace_shapes(self._max_decode_rows, splits)
            )
        )

    def _forward_compressed(self, q_TNH, cache, metadata, role, pool, output):
        """Run the layer on the runtime of this step's size, creating it on
        first use."""
        from aleph_alpha_inference.kernels.serving_compressed import (
            ServingCompressedLayer,
        )

        count = metadata.num_actual_tokens
        decode = metadata.max_query_len == 1
        # Prefill spans are powers of two, so that few runtimes cover them.
        span = (
            self._decode_span_limit
            if decode
            else max(16384, 1 << (metadata.max_seq_len - 1).bit_length())
        )
        runtimes = self._decode_runtimes if decode else self._prefill_runtimes
        runtime_key = (_workspace_owner(), count, span, cache.storage.data_ptr())
        runtime = runtimes.get(runtime_key)
        if runtime is None:
            # Eager decode under DP runs every distinct batch size whenever any
            # rank prefills. Runtimes view one bounded arena per lane instead
            # of allocating per size, which leaked GPU memory during serving.
            scratch, source_pool = self._compressed_routing_workspace(
                q_TNH.device, decode
            )
            runtime = ServingCompressedLayer(
                cache,
                metadata.block_table,
                count,
                span,
                self._topk,
                decode,
                scratch=scratch,
                source_pool=source_pool,
                allow_staging=self._allow_prefill_staging,
                pool_tokens=self._pool_tokens,
            )
            runtimes[runtime_key] = runtime
            # Runtimes own no device memory, so evicting a captured size is safe.
            while len(runtimes) > (64 if decode else 4):
                runtimes.popitem(last=False)
        runtimes.move_to_end(runtime_key)
        runtime.table = metadata.block_table
        attention_workspace = self._compressed_decode_workspace() if decode else None
        runtime.run(
            q_TNH,
            metadata.query_start_loc,
            metadata.seq_lens,
            role=role,
            pool=pool,
            out=output[:count, None],
            attention_workspace=attention_workspace,
            bounds=metadata.row_bounds,
            query_starts=metadata.query_starts,
            min_context=metadata.min_context,
        )
        if role == "source":
            self.routing_pool = runtime.pool
        return output

    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
        output_scale=None,
        output_block_scale=None,
    ):
        # Re-read the live context: breakable CUDA graphs replay this call with
        # its capture-time arguments. KV-sharing layers read their source's.
        owner = getattr(layer, "kv_source_name", layer.layer_name)
        metadata, *_ = get_attention_context(owner)
        if metadata is None:
            if query.is_cuda:
                self._reserve_scratch(query.device)
            return output.zero_()
        if output_scale is not None or output_block_scale is not None:
            raise ValueError("Block8 attention produces BF16 output")
        cache = self._cache_view(owner)
        count = metadata.num_actual_tokens
        q_TNH = query[:count, None]
        role = "source" if self.sparse_role == "indexer" else "reindex"
        pool = None if role == "source" else self.source_attention.impl.routing_pool
        return self._forward_compressed(q_TNH, cache, metadata, role, pool, output)
