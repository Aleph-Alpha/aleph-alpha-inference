# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""Prefill launch skips leave the served results bit-identical.

Two skips depend on what a step tells the layers:
- The short scan is skipped when every row's context exceeds the pool
  (`min_context`).
- Layers that share a step (one `bounds` list) and a cache skip restaging a
  context that the staging buffer already holds.

Each case runs a source and a reindex layer on one cache twice: once with a
fresh `bounds` list per layer and no context bound, once with a shared list
and the true shortest context. Outputs and routes must be identical, and the
skipping run must launch fewer kernels.
"""

import itertools

import pytest
import torch

from test_serving_stale_scratch import make_cache

pytestmark = pytest.mark.gpu

TOPK = 2048
POOL_TOKENS = 16384
# (prefix, queries) per request; requests of >= 21 queries attend over union
# tiles and stage their context.
CASES = {
    # Every row past the pool. Two tiled requests alternate in the staging
    # buffer, so the reindex layer restages both.
    "past-pool": [(16384 + 64, 300), (20000, 64)],
    # Every row past the pool, one tiled request: the reindex layer reuses it.
    "past-pool-one-tile": [(16384 + 64, 300), (20000, 1)],
    # Rows on both sides of the pool; the short scan must run.
    "both-sides": [(8, 1), (16384 - 150, 300)],
}
# Expected (short scan, context staging) launches of the skipping run, given
# one chunk of rows per layer. The baseline launches 2 short scans and stages
# every tiled request twice.
EXPECTED = {
    "past-pool": (0, 4),
    "past-pool-one-tile": (0, 1),
    "both-sides": (2, 1),
}
_owners = itertools.count()


@pytest.fixture
def fresh_arenas():
    from aleph_alpha_inference.kernels import serving_compressed

    yield
    for cached in (
        serving_compressed.routing_workspace,
        serving_compressed.source_pool,
    ):
        cached.cache_clear()
    torch.cuda.empty_cache()


class LaunchCounter:
    """Counts launches of the kernels that the skips remove."""

    def __init__(self, monkeypatch):
        from aleph_alpha_inference.kernels import prefill_staging, serving_compressed
        from aleph_alpha_inference.kernels.summary_scan import GroupedShortScan

        self.short = self.staging = 0
        routing_launch = serving_compressed._COMPILED.launch
        staging_launch = prefill_staging._COMPILED.launch

        def count_routing(executable, *args, **kwargs):
            self.short += isinstance(executable, GroupedShortScan)
            return routing_launch(executable, *args, **kwargs)

        def count_staging(*args, **kwargs):
            self.staging += 1
            return staging_launch(*args, **kwargs)

        monkeypatch.setattr(serving_compressed._COMPILED, "launch", count_routing)
        monkeypatch.setattr(prefill_staging._COMPILED, "launch", count_staging)

    def take(self):
        """Launch counts since the last call, as (short scans, stagings)."""
        counts = self.short, self.staging
        self.short = self.staging = 0
        return counts


def run_layers(cache, table, queries, cu, ends, starts, span, *, skip):
    """Source then reindex on fresh served arenas, as `Block8Impl`
    builds them. With `skip`, the layers share a step and know the shortest
    context. Returns (out, route, used, pool) per layer."""
    from aleph_alpha_inference.kernels.serving_compressed import (
        routing_workspace,
        ServingCompressedLayer,
        source_pool,
    )
    from aleph_alpha_inference.kernels.vllm_backend import min_row_context

    rows = queries[0].shape[0]
    owner = ("launch-skips", next(_owners))
    scratch = routing_workspace(
        queries[0].device, owner, rows, span, TOPK, False, POOL_TOKENS
    )
    step = []
    min_context = min_row_context(cu.cpu(), ends.cpu()) if skip else 0
    results, pool = [], None
    for role, q in zip(("source", "reindex"), queries):
        layer = ServingCompressedLayer(
            cache,
            table,
            rows,
            span,
            TOPK,
            False,
            scratch=scratch,
            source_pool=source_pool(q.device, owner, "source", rows, POOL_TOKENS)
            if role == "source"
            else None,
            pool_tokens=POOL_TOKENS,
        )
        out = torch.full_like(q, float("nan"))
        layer.run(
            q,
            cu,
            ends,
            role=role,
            pool=pool,
            out=out,
            bounds=step if skip else [],
            query_starts=tuple(starts),
            min_context=min_context,
        )
        if role == "source":
            pool = layer.pool
        results.append(
            tuple(x.clone() for x in (out, layer.route, layer.used, layer.pool))
        )
    torch.cuda.synchronize()
    return results


@pytest.mark.parametrize("case", list(CASES))
def test_launch_skips_are_exact(case, fresh_arenas, monkeypatch):
    segments = CASES[case]
    lengths = [p + n for p, n in segments]
    starts = [0]
    for _, n in segments:
        starts.append(starts[-1] + n)
    cache, table, _ = make_cache(16, lengths)
    rows = starts[-1]
    g = torch.Generator(device="cuda").manual_seed(5)
    queries = [
        torch.randn(rows, 1, 48, 128, device="cuda", dtype=torch.bfloat16, generator=g)
        for _ in range(2)
    ]
    cu = torch.tensor(starts, device="cuda", dtype=torch.int32)
    ends = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    span = max(16384, 1 << (max(lengths) - 1).bit_length())
    tiled = sum(n >= 21 for _, n in segments)

    counter = LaunchCounter(monkeypatch)
    baseline = run_layers(cache, table, queries, cu, ends, starts, span, skip=False)
    assert counter.take() == (2, 2 * tiled)
    skipping = run_layers(cache, table, queries, cu, ends, starts, span, skip=True)
    assert counter.take() == EXPECTED[case]

    for role, expected, actual in zip(("source", "reindex"), baseline, skipping):
        _, _, used, _ = expected
        for name, a, b in zip(("out", "route", "used", "pool"), expected, actual):
            if name == "route":
                # Slots past `used` are not part of the route.
                live = torch.arange(a.shape[1], device="cuda") < used[:, None]
                a, b = a[live], b[live]
            torch.testing.assert_close(b, a, rtol=0, atol=0, msg=f"{role} {name}")


def test_new_step_restages_written_cache():
    from aleph_alpha_inference.kernels.prefill_staging import stage_context

    lengths = [4096, 1024]
    cache, table, _ = make_cache(16, lengths)
    ends = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    first = [x.clone() for x in stage_context(cache, table, ends, 0, 4096, step=[])]

    # The next step writes new keys and values over the first two pages.
    g = torch.Generator(device="cuda").manual_seed(7)
    k, v = (
        torch.randn(256, 4, 128, device="cuda", dtype=torch.bfloat16, generator=g)
        for _ in range(2)
    )
    ids = torch.arange(256, device="cuda")
    cache.append(k, v, table[0, ids // 128].long() * 128 + ids % 128)

    # Restage under the new step before the step-less call, which clears the
    # record.
    restaged = [x.clone() for x in stage_context(cache, table, ends, 0, 4096, step=[])]
    fresh = stage_context(cache, table, ends, 0, 4096)
    assert not torch.equal(restaged[0][:256], first[0][:256])
    for name, a, b in zip(("k", "v"), fresh, restaged):
        torch.testing.assert_close(b, a, rtol=0, atol=0, msg=name)
