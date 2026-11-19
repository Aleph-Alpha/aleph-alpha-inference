# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
import pytest
import torch
from aleph_alpha_inference.kernels.serving_compressed import row_bounds


def _reference(cu, seq_lens, rows):
    """Row bounds by a scan over the requests."""
    requests, lengths = [0] * rows, [0] * rows
    for request, (start, end) in enumerate(zip(cu, cu[1:])):
        for row in range(start, min(end, rows)):
            requests[row] = request
            lengths[row] = seq_lens[request] - (end - row - 1)
    return requests, lengths


@pytest.mark.parametrize(
    "queries, padding",
    [
        # Decode: one query per request.
        ([1] * 7, 0),
        # A captured decode graph replayed with fewer requests than rows.
        ([1] * 5, 3),
        # Prefill with uneven requests.
        ([4, 1, 9, 2], 0),
        # Empty requests at the start, in the middle and at the end.
        ([0, 3, 0, 0, 2, 1, 0], 0),
        ([0, 0, 5], 2),
        ([6], 0),
        # Only padding.
        ([0, 0], 4),
    ],
)
def test_row_bounds_match_request_scan(queries, padding):
    cu = [0]
    for count in queries:
        cu.append(cu[-1] + count)
    rows = cu[-1] + padding
    seq_lens = [100 * (i + 1) + count for i, count in enumerate(queries)]
    requests = torch.full((rows,), -1, dtype=torch.int32)
    lengths = torch.full((rows,), -1, dtype=torch.int32)
    row_bounds(
        torch.tensor(cu, dtype=torch.int32),
        torch.tensor(seq_lens, dtype=torch.int32),
        requests,
        lengths,
    )
    expected_requests, expected_lengths = _reference(cu, seq_lens, rows)
    assert requests.tolist() == expected_requests
    assert lengths.tolist() == expected_lengths


@pytest.mark.gpu
def test_row_bounds_follow_new_batches_in_a_replayed_graph():
    """A captured decode graph reads the bounds of each replayed batch."""
    rows = 8
    cu = torch.arange(rows + 1, device="cuda", dtype=torch.int32)
    seq_lens = torch.full((rows,), 10, device="cuda", dtype=torch.int32)
    requests = torch.empty(rows, device="cuda", dtype=torch.int32)
    lengths = torch.empty_like(requests)
    row_bounds(cu, seq_lens, requests, lengths)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        row_bounds(cu, seq_lens, requests, lengths)
    # vLLM refreshes the captured tensors in place: five live requests, the
    # other rows are padding.
    cu.copy_(torch.arange(rows + 1).clamp(max=5))
    seq_lens.copy_(torch.arange(rows) * 100 + 7)
    graph.replay()
    live = torch.arange(rows, device="cuda") < 5
    torch.testing.assert_close(
        requests, (torch.arange(rows, device="cuda") * live).int(), rtol=0, atol=0
    )
    torch.testing.assert_close(lengths, (seq_lens * live).int(), rtol=0, atol=0)
