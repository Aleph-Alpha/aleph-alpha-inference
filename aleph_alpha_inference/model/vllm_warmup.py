# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""Compile every Kolibri prefill attention executable before serving.

vLLM's startup dummy batches are short, so the first long prompt used to
compile CuTe executables in the middle of a DP/EP step while the other ranks
spun in the MoE all-to-all. Prefill executables specialize on a bounded set of
dispatch dimensions: the routing span bucket (16k .. max_model_len, powers of
two), single- or multi-request staging, and FA4's query-tile and split choices.
Running the attention layers once per point of that set, on the null KV block,
compiles them all before vLLM's JIT monitor starts.

The batches run attention only: no KV is written, scratch comes from the arenas
reserved during profiling, and nothing is kept but the compiled executables.
"""

import time

import torch
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.v1.worker.gpu.input_batch import InputBatch, set_dummy_context

from aleph_alpha_inference.kernels.cute_jit import executable_counts

logger = init_logger(f"vllm.{__name__}")

# The routing span bucket of a prefill (vllm_backend._forward_compressed).
MIN_SPAN = 16384


def warmup_batches(max_model_len, max_num_reqs, max_num_tokens):
    """(requests, query tokens per request, sequence length) of each batch."""
    batches = []
    span = MIN_SPAN
    while span // 2 < max_model_len:
        seq_len = min(span, max_model_len)
        # One request stages its context; several read paged KV.
        batches += [(1, 2, seq_len), (2, 2, seq_len)]
        span *= 2
    # SWA FA4 picks its Q tile stage, persistence and KV splits from the query
    # length, request count and token count (eager batches of <= 1024 tokens
    # carry vLLM's CUDA-graph split bound).
    for requests, query, context in (
        (1, 16, 0),
        (1, 64, 0),
        (1, 2048, 0),
        (1, 2, 4096),
        (1, 64, 4096),
        (1, 2048, 4096),
        (64, 2, 4096),
        (64, 16, 4096),
        (64, 32, 4096),
    ):
        batches.append((requests, query, query + context))
    return [
        (requests, query, min(seq_len, max_model_len))
        for requests, query, seq_len in batches
        if requests <= max_num_reqs
        and requests * query <= max_num_tokens
        and query <= max_model_len
    ]


def _layers(context):
    layers = [layer for layer in context.values() if isinstance(layer, Attention)]
    # Consumers read the routing pool of their source: run sources first.
    return sorted(
        layers, key=lambda layer: getattr(layer.impl, "sparse_role", "") != "indexer"
    )


@torch.inference_mode()
def warm_up_attention(state, prepare, input_buffers, block_tables, kv_cache_config):
    """Run each warmup batch through every attention layer; `prepare` builds
    the batch's attention metadata as a real step would."""
    config = state.vllm_config
    context = config.compilation_config.static_forward_context
    layers = _layers(context)
    dp = config.parallel_config.data_parallel_size
    start, compiles = time.perf_counter(), executable_counts()
    for requests, query, seq_len in warmup_batches(
        state.max_model_len,
        config.scheduler_config.max_num_seqs,
        config.scheduler_config.max_num_batched_tokens,
    ):
        tokens = requests * query
        batch = InputBatch.make_dummy(
            requests, tokens, input_buffers, max_query_len=query
        )
        tables = block_tables.get_dummy_block_tables(requests)
        set_dummy_context(
            batch,
            block_tables,
            seq_len - query,
            kv_cache_config.num_blocks,
            state.max_model_len,
            input_block_tables=tables,
        )
        # Read only the null block: the context was never written.
        for table in tables:
            table.zero_()
        metadata = prepare(batch, tables, block_tables.get_dummy_slot_mappings(tokens))
        with set_forward_context(
            metadata,
            config,
            num_tokens=tokens,
            num_tokens_across_dp=torch.full((dp,), tokens, dtype=torch.int32),
        ):
            for layer in layers:
                owner = getattr(layer, "kv_source_name", layer.layer_name)
                q = torch.zeros(
                    tokens,
                    layer.num_heads,
                    layer.head_size,
                    dtype=config.model_config.dtype,
                    device=state.device,
                )
                kv = (
                    None
                    if owner != layer.layer_name
                    else torch.zeros(
                        tokens,
                        layer.num_kv_heads,
                        layer.head_size,
                        dtype=q.dtype,
                        device=q.device,
                    )
                )
                out = torch.empty(
                    tokens,
                    layer.num_heads,
                    layer.head_size_v,
                    dtype=q.dtype,
                    device=q.device,
                )
                layer.impl.forward(
                    layer, q, kv, kv, context[owner].kv_cache, metadata[owner], out
                )
    torch.accelerator.synchronize()
    # Return the batches' transient buffers; scratch arenas stay reserved.
    torch.accelerator.empty_cache()
    after = executable_counts()
    logger.info(
        "Kolibri attention warmup compiled %d executables in %.1f s; "
        "executables per cache: %s",
        sum(after.values()) - sum(compiles.values()),
        time.perf_counter() - start,
        after,
    )
