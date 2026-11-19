# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""SWA metadata rebases each prefill's block table at its replay start's tile."""

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import torch
from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import SlidingWindowSpec

from tensor_asserts import assert_equal

# The tests import `vllm_swa` themselves: it needs vLLM's CUDA flash attention
# extensions, which the CPU-only CI runner lacks, so importing it here would
# fail collection there.

if TYPE_CHECKING:
    from aleph_alpha_inference.model.vllm_swa import CuteSWAMetadata


def build(
    page: int,
    replay_start: list[int],
    max_query_len: int,
) -> tuple["CuteSWAMetadata", torch.Tensor, torch.Tensor]:
    """Build the metadata of three requests, each `max_query_len` tokens long."""
    from aleph_alpha_inference.model.vllm_swa import CuteSWAMetadataBuilder

    config = VllmConfig()
    config.model_config = SimpleNamespace(
        get_num_attention_heads=lambda _: 48,
        hf_config=SimpleNamespace(model_type="kolibri1"),
        hf_text_config=SimpleNamespace(),
        rswa_window=None,
        is_mm_prefix_lm=False,
    )
    spec = SlidingWindowSpec(
        block_size=page,
        num_kv_heads=4,
        head_size=128,
        dtype=torch.bfloat16,
        sliding_window=513,
    )
    builder = CuteSWAMetadataBuilder(spec, [], config, torch.device("cuda"))
    table = torch.arange(3 * 256, device="cuda", dtype=torch.int32).view(3, 256)
    seq_lens = torch.tensor([2048, 2049, 2050], device="cuda", dtype=torch.int32)
    query_start_loc = torch.tensor(
        [0, max_query_len, 2 * max_query_len, 3 * max_query_len],
        device="cuda",
        dtype=torch.int32,
    )
    common = SimpleNamespace(
        num_reqs=3,
        num_actual_tokens=3 * max_query_len,
        max_query_len=max_query_len,
        max_seq_len=2050,
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.cpu(),
        seq_lens=seq_lens,
        block_table_tensor=table,
        slot_mapping=torch.arange(3 * max_query_len, device="cuda"),
        causal=True,
        mm_req_doc_ranges=None,
        rswa_prefix_lens=None,
    )
    start = torch.tensor(replay_start, device="cuda", dtype=torch.int32)
    return builder.build(0, common, replay_start=start), table, seq_lens


@pytest.mark.gpu
@pytest.mark.parametrize("page", [16, 64, 128])
def test_prefill_rebase_keeps_tile_origin(page):
    metadata, table, seq_lens = build(page, [1535, 1537, 0], 3)
    # Origins round down to the 128-key FA4 tile holding the replay start.
    origins = torch.tensor([1408, 1536, 0], device="cuda", dtype=torch.int32)
    torch.testing.assert_close(
        metadata.replay_floor, torch.tensor([127, 1, 0], device="cuda").int()
    )
    assert_equal(metadata.seq_lens, seq_lens - origins)
    for row, origin in enumerate(origins.tolist()):
        shift = origin // page
        torch.testing.assert_close(
            metadata.block_table[row, : 256 - shift], table[row, shift:]
        )
    # The kernel keeps the batch's bound, which its split policy reads.
    assert metadata.max_seq_len == 2050


@pytest.mark.gpu
def test_decode_ignores_replay_starts():
    metadata, table, seq_lens = build(64, [0, 0, 0], 1)
    assert metadata.replay_floor is None
    assert metadata.block_table is table
    assert metadata.seq_lens is seq_lens
