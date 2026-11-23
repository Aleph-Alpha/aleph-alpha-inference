# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""Reusing cache pages outside a sliding window must not change attention."""

import inspect

import pytest
import torch
from vllm.v1.worker.gpu.block_table import BlockTables

from tensor_asserts import assert_equal

# The tests import `fa_utils.flash_attn_varlen_func` themselves: it needs vLLM's
# CUDA flash attention extensions, which the CPU-only CI runner lacks, so
# importing it here would fail collection there.


@pytest.mark.gpu
@pytest.mark.parametrize("page", [16, 64])
def test_reused_out_of_window_pages_cannot_poison_attention(page):
    # The test exercises vLLM #60767, which added sliding_window_reach. The
    # plugin requires a vLLM that has it.
    assert "sliding_window_reach" in inspect.signature(BlockTables).parameters
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

    torch.manual_seed(17)
    device = torch.device("cuda")
    length, window = 710, 513
    pages = (length + page - 1) // page
    # sliding_window_reach and num_computed_tokens come from vLLM #60767.
    tables = BlockTables(
        block_sizes=[page, page],
        max_num_reqs=2,
        max_num_batched_tokens=1,
        max_num_blocks_per_group=[pages, pages],
        device=device,
        sliding_window_reach=[window - 1, 0],
    )
    ids = list(range(1, pages + 1))
    tables.append_block_ids(1, (ids, ids), overwrite=True)
    tables.apply_staged_writes()
    mapping = torch.tensor([1], device=device, dtype=torch.int32)
    computed = torch.tensor([0, length - 1], device=device, dtype=torch.int32)
    persistent = tuple(t.gpu.clone() for t in tables.block_tables)
    sliding, global_table = tables.gather_block_tables(
        mapping, num_reqs_padded=1, num_computed_tokens=computed
    )
    assert_equal(global_table[0, :pages], persistent[1][1, :pages])

    cache = torch.randn(pages + 1, 4, page, 256, device=device, dtype=torch.bfloat16)
    cache[0].zero_()
    key, value = cache.transpose(1, 2).split(128, dim=-1)
    query = torch.randn(1, 48, 128, device=device, dtype=torch.bfloat16)
    output = torch.empty_like(query)

    def attend():
        flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            out=output,
            cu_seqlens_q=torch.tensor([0, 1], device=device, dtype=torch.int32),
            max_seqlen_q=1,
            seqused_k=torch.tensor([length], device=device, dtype=torch.int32),
            max_seqlen_k=length,
            softmax_scale=128**-0.5,
            causal=True,
            window_size=(window - 1, 0),
            block_table=sliding,
            fa_version=4,
            num_splits=1,
        )
        return output.clone()

    expected = attend()
    # Simulate another cache group reusing every wholly out-of-window page.
    # These bytes are masked mathematically but share an attention load tile.
    freed = (length - window) // page
    cache[1 : freed + 1].fill_(float("nan"))
    actual = attend()
    assert_equal(actual, expected)
    for table, before in zip(tables.block_tables, persistent):
        assert_equal(table.gpu, before)
