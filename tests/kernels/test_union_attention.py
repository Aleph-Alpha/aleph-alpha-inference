# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""Union tiles hold exactly each tile's keys; FA4 over them matches attention."""

import pytest
import torch

pytestmark = pytest.mark.gpu


def _routes(rows, width, span, used_min=1):
    g = torch.Generator(device="cuda").manual_seed(9)
    route = torch.randint(0, span, (rows, width), device="cuda", generator=g)
    # Distinct keys per row, like the selector's.
    route = torch.stack(
        [torch.randperm(span, device="cuda", generator=g)[:width] for _ in range(rows)]
    )
    used = torch.randint(used_min, width + 1, (rows,), device="cuda", generator=g).int()
    return route.int().contiguous(), used


def _check_tiles(route, used, tiles, tile_rows, limit, gate):
    keys, count, fallback, qrows, mask = tiles
    rows = route.shape[0]
    unions = []
    for t in range((rows + tile_rows - 1) // tile_rows):
        rs = range(t * tile_rows, min(rows, (t + 1) * tile_rows))
        sets = [set(route[r, : int(used[r])].tolist()) for r in rs]
        unions.append((sets, sorted(set().union(*sets))))
    gated = sum(len(u) for _, u in unions) > gate * len(unions)
    built = 0
    for t, (sets, union) in enumerate(unions):
        if gated or len(union) > limit:
            assert int(fallback[t]) == 1 and int(count[t]) == int(qrows[t]) == 0
            continue
        built += 1
        assert int(fallback[t]) == 0 and int(count[t]) == len(union)
        assert int(qrows[t]) == len(sets)
        assert keys[t, : len(union)].tolist() == union
        bits = mask[t].view(torch.uint8).view(-1, mask.shape[2] * 4)
        bits = torch.stack([(bits >> i) & 1 for i in range(8)], -1).flatten(1)
        for j, row in enumerate(sets):
            want = torch.tensor([u in row for u in union], device="cuda")
            assert torch.equal(bits[j, : len(union)].bool(), want)
    return built


@pytest.mark.parametrize("span,width", [(4096, 300), (70000, 2184)])
def test_union_tiles_match_sets(span, width):
    from aleph_alpha_inference.kernels import union_attention as ua

    rows = 50
    route, used = _routes(rows, width, span)
    used[3] = 0
    ws = ua.union_workspace(rows, "cuda")
    tiles = ua.union_tiles(route, used, span, ws)
    _check_tiles(route, used, tiles, ua.TILE_ROWS, ua.UNION_LIMIT, ua.union_gate())


def test_spill_flags_tiles_over_the_limit():
    from aleph_alpha_inference.kernels import union_attention as ua

    ws = ua.union_workspace(42, "cuda")

    def tiles(width):
        route, used = _routes(42, width, 160000, used_min=width)
        return ua.union_tiles(route, used, 160000, ws)

    # 21 rows of 1000 distinct keys exceed the limit of a tile.
    _, count, fallback, qrows, _ = tiles(1000)
    assert fallback.tolist() == [1, 1] and count.tolist() == qrows.tolist() == [0, 0]
    # Under the limit but over the gate on average: the whole request spills.
    _, count, fallback, qrows, _ = tiles(600)
    assert fallback.tolist() == [1, 1] and count.tolist() == [0, 0]
    _, count, fallback, qrows, _ = tiles(400)
    assert fallback.tolist() == [0, 0] and min(count.tolist()) > 4000


def test_union_attention_matches_reference():
    from aleph_alpha_inference.kernels import union_attention as ua

    rows, span, width = 45, 3000, 640
    route, used = _routes(rows, width, span)
    q = torch.randn(rows, 48, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(span, 1, 4, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    ws = ua.union_workspace(rows, "cuda")
    out = ua.union_attention(q, k, v, ua.union_tiles(route, used, 4096, ws))
    for r in range(rows):
        ix = route[r, : int(used[r])].long()
        s = torch.einsum(
            "hjd,nhd->hjn", q[r].float().view(4, 12, 128), k[ix, 0].float()
        )
        o = torch.einsum(
            "hjn,nhd->hjd", (s * 128**-0.5).softmax(-1), v[ix, 0].float()
        )
        torch.testing.assert_close(
            out[r].float(), o.reshape(48, 128), atol=1e-2, rtol=1e-2
        )
