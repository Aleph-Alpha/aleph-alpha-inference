# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""Cache fields in vLLM-planned pages serve exactly what contiguous fields do.

The reference keeps every field in its own contiguous allocation.
The actual caches are views of vLLM's plan for four KV owners beside SWA pages:
each owner's page is one record whose fields start on 512-byte boundaries.
"""

from types import SimpleNamespace

import pytest
import torch

from sparse_hf_config import sparse_hf_config


pytestmark = pytest.mark.gpu

PAGE = 128
OWNERS = 4
LENGTHS = [257, 644, 645, 17017]


def _planned_caches(monkeypatch, blocks):
    import aleph_alpha_inference.kernels.vllm_backend as backend
    from vllm.v1.core.kv_cache_planning import get_kv_cache_config_from_groups
    from vllm.v1.kv_cache_interface import (
        create_kv_cache_views,
        FullAttentionSpec,
        KVCacheGroupSpec,
        KVCacheLayout,
        SlidingWindowSpec,
        UniformTypeKVCacheSpecs,
    )

    cache_type = backend.SummaryCache
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            max_model_len=32768,
            hf_text_config=sparse_hf_config(index_compression=16, index_topk=2048),
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=1, max_num_batched_tokens=1),
        compilation_config=SimpleNamespace(max_cudagraph_capture_size=None),
        parallel_config=SimpleNamespace(use_ubatching=False),
        cache_config=SimpleNamespace(
            block_size=PAGE,
            get_resolved_kv_cache_layout=lambda: KVCacheLayout.BLHNC,
            num_gpu_blocks_override=blocks,
            prefix_cache_retention_interval=None,
        ),
        attention_config=SimpleNamespace(hisparse_config=None),
    )
    monkeypatch.setattr(backend, "get_current_vllm_config", lambda: config)
    kv = backend.serving_backend(16).customize_spec(
        FullAttentionSpec(
            block_size=PAGE, num_kv_heads=4, head_size=128, dtype=torch.bfloat16
        )
    )
    owners = {f"{owner}": kv for owner in range(OWNERS)}
    swa = SlidingWindowSpec(
        block_size=64,
        num_kv_heads=4,
        head_size=128,
        dtype=torch.bfloat16,
        sliding_window=513,
    )
    groups = [
        KVCacheGroupSpec(list(specs), UniformTypeKVCacheSpecs.from_specs(specs))
        for specs in ({f"swa.{n}": swa for n in range(2)}, owners)
    ]
    stride = OWNERS * backend.record_bytes(cache_type, PAGE)
    planned = get_kv_cache_config_from_groups(config, groups, blocks * stride)
    # The production plan: owners' pages set the block stride beside two SWA pages.
    assert {t.block_stride for t in planned.kv_cache_tensors} == {stride}
    raw = torch.zeros(blocks * stride, device="cuda", dtype=torch.uint8)
    (tensor,) = planned.kv_cache_tensors[1:]
    views = create_kv_cache_views(raw, kv, blocks, KVCacheLayout.BLHNC, tensor)
    layers = {
        name: SimpleNamespace(kv_cache=view) for name, view in zip(tensor.layers, views)
    }
    monkeypatch.setattr(
        backend,
        "get_forward_context",
        lambda: SimpleNamespace(no_compile_layers=layers),
    )
    # Each owner's attention slices its fields from its record, as in serving.
    return [
        backend.Block8Impl(48, 128, 128**-0.5, num_kv_heads=4)._cache_view(f"{owner}")
        for owner in range(OWNERS)
    ]


def _record_cache(cache_type, blocks):
    return cache_type(
        *(
            torch.zeros(blocks, width, device="cuda", dtype=torch.uint8)
            for width in cache_type.page_bytes(PAGE)
        ),
        page=PAGE,
    )


def _append(cache, table, k, v):
    offset = 0
    chunks = torch.Generator().manual_seed(29)
    for row, length in enumerate(LENGTHS):
        start = 0
        while start < length:
            # Ragged chunks cross block and page boundaries.
            count = min(
                int(torch.randint(1, 3000, (1,), generator=chunks)), length - start
            )
            ids = torch.arange(start, start + count, device="cuda")
            slots = table[row, ids // PAGE].long() * PAGE + ids % PAGE
            rows = slice(offset + start, offset + start + count)
            cache.append(k[rows], v[rows], slots)
            start += count
        offset += length
    lengths = torch.tensor(LENGTHS, device="cuda", dtype=torch.int32) + 1
    decode_k, decode_v = k[offset:], v[offset:]
    slots = table.gather(1, (lengths[:, None] - 1).long() // PAGE)[:, 0].long()
    cache.append(decode_k, decode_v, slots * PAGE + (lengths - 1) % PAGE)
    return lengths


def _outputs(cache, table, lengths, queries):
    from aleph_alpha_inference.kernels.prefill_staging import stage_context

    from serving_arena import (
        decode_scratch,
        new_owner,
        served_layer,
    )

    out = {}
    pages = (lengths + PAGE - 1) // PAGE
    for row, count in enumerate(pages.tolist()):
        ids = table[row, :count].long()
        out[f"fields.{row}"] = [
            tensor[ids, :width]
            for tensor, width in zip(cache.tensors, cache.page_bytes(PAGE))
        ]
    one, one_length = table[3:4], lengths[3:4]
    out["context"] = stage_context(cache, one, one_length, 0, int(one_length))
    for decode in (True, False):
        for name, rows_table, rows_lengths in (
            ("all", table, lengths),
            ("one", one, one_length),
        ):
            count = rows_table.shape[0]
            bounds = torch.arange(count + 1, device="cuda", dtype=torch.int32)
            q = queries[:count]
            pool, owner = None, new_owner()
            for role in ("source", "reindex"):
                layer = served_layer(
                    cache,
                    rows_table,
                    count,
                    int(rows_lengths.max()),
                    2048,
                    decode,
                    owner=owner,
                    source=role == "source",
                )
                result = torch.empty_like(q)
                layer.run(
                    q,
                    bounds,
                    rows_lengths,
                    role=role,
                    pool=pool,
                    out=result,
                    bounds=[],
                    query_starts=tuple(range(count + 1)),
                    attention_workspace=decode_scratch(layer) if decode else None,
                )
                pool = layer.pool
                out[f"compressed.{decode}.{name}.{role}"] = (
                    result,
                    layer.route,
                    layer.used,
                )
    return _cloned(out)


def _cloned(outputs):
    # Outputs may view scratch that the other cache's runs overwrite.
    if isinstance(outputs, torch.Tensor):
        return outputs.clone()
    if isinstance(outputs, dict):
        return {key: _cloned(value) for key, value in outputs.items()}
    return [_cloned(value) for value in outputs]


def test_planned_cache_pages_match_one_record_per_page(monkeypatch):
    import vllm.vllm_flash_attn  # noqa: F401 - Install the serving FA4 alias.
    from aleph_alpha_inference.kernels.summary_cache import SummaryCache

    torch.manual_seed(611)
    pages = sum((n + PAGE) // PAGE for n in LENGTHS)
    blocks = pages + 9
    table = torch.full(
        (len(LENGTHS), max((n + PAGE) // PAGE for n in LENGTHS)),
        -1,
        device="cuda",
        dtype=torch.int32,
    )
    ids = torch.randperm(blocks - 1, device="cuda", dtype=torch.int32) + 1
    start = 0
    for row, length in enumerate(LENGTHS):
        count = (length + PAGE) // PAGE
        table[row, :count] = ids[start : start + count]
        start += count
    k = torch.randn(
        sum(LENGTHS) + len(LENGTHS), 4, 128, device="cuda", dtype=torch.bfloat16
    )
    v = torch.randn_like(k)
    queries = torch.randn(len(LENGTHS), 1, 48, 128, device="cuda", dtype=torch.bfloat16)
    reference = _record_cache(SummaryCache, blocks)
    planned = _planned_caches(monkeypatch, blocks)
    # Fill another owner first: owners share every block of the allocation.
    _append(planned[0], table, torch.randn_like(k), torch.randn_like(v))
    expected_lengths = _append(reference, table, k, v)
    actual_lengths = _append(planned[2], table, k, v)
    expected = _outputs(reference, table, expected_lengths, queries)
    actual = _outputs(planned[2], table, actual_lengths, queries)
    assert expected.keys() == actual.keys()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
