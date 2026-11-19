# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""Fused sigmoid_logit_add router (``aleph_alpha_inference.routing``).

The fused Triton kernel replaces ``sigmoid_logit_add_routing`` on CUDA, so it
must reproduce it bit for bit: same expert ids in the same slot order and the
same float32 weights. The reference is the ``torch.compile``d function the
model ran before, not eager torch: eager ``torch.sigmoid`` differs from the
inductor sigmoid by up to ~12 ulp.

Exact score ties are the one case where "same order" is not well defined:
torch.topk on CUDA keeps the lowest-index tied expert at the top-k boundary
but orders tied experts arbitrarily inside the top-k. For ties the tests
therefore compare the selected set (with per-expert weights) against torch
and the slot order against the kernel's own rule (score descending, lowest
index first).
"""

import pytest
import torch
import torch._dynamo

pytestmark = pytest.mark.gpu

REAL_EXPERTS, REAL_TOP_K = 384, 6  # Kolibri 1


@pytest.fixture(autouse=True)
def _cuda_and_compile_limits():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    # The reference is torch.compiled and specialises on topk/renormalize
    # and dtypes. Past the recompile limit dynamo silently falls back to
    # eager, whose sigmoid is not bit-identical; fail loudly instead.
    with torch._dynamo.config.patch(
        recompile_limit=256, fail_on_recompile_limit_hit=True
    ):
        yield


def _reference(logits, bias, topk, renormalize):
    from aleph_alpha_inference.kolibri1 import sigmoid_logit_add_routing

    return sigmoid_logit_add_routing(
        hidden_states=None,
        gating_output=logits,
        topk=topk,
        renormalize=renormalize,
        e_score_correction_bias=bias,
    )


def _fused(logits, bias, topk, renormalize):
    from aleph_alpha_inference.routing import fused_sigmoid_logit_add_topk

    return fused_sigmoid_logit_add_topk(logits, bias, topk, renormalize)


def _make_inputs(num_tokens, num_experts, distribution, seed=0):
    """Router logits and expert bias for the regimes the model can hit."""
    g = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, device="cuda", generator=g)

    if distribution == "randn":
        logits, bias = randn(num_tokens, num_experts) * 3, randn(num_experts) * 5
    elif distribution == "bias_dominated":
        # Trained checkpoints: bias up to ~20, router logits much smaller, so
        # many tokens share experts and scores sit close together.
        logits, bias = randn(num_tokens, num_experts) * 0.3, randn(num_experts) * 10
    elif distribution == "zero_bias":
        # Freshly initialised bias (torch.zeros in Kolibri1SparseMoeBlock).
        logits = randn(num_tokens, num_experts)
        bias = torch.zeros(num_experts, device="cuda")
    elif distribution == "saturated":
        # sigmoid saturates to 1.0 and underflows into denormals / 0.
        logits, bias = randn(num_tokens, num_experts) * 60, randn(num_experts) * 5
    else:
        raise ValueError(distribution)
    return logits, bias


def _assert_well_formed(weights, ids, num_tokens, num_experts, topk):
    assert weights.shape == ids.shape == (num_tokens, topk)
    assert weights.dtype == torch.float32 and ids.dtype == torch.int32
    assert weights.is_contiguous() and ids.is_contiguous()
    if num_tokens == 0:
        return
    assert ids.min() >= 0 and ids.max() < num_experts, "expert id out of range"
    sorted_ids = ids.sort(dim=-1).values
    dup_rows = (sorted_ids[:, 1:] == sorted_ids[:, :-1]).any(dim=-1)
    assert not dup_rows.any(), (
        f"duplicate expert ids in {int(dup_rows.sum())} rows, "
        f"e.g. {ids[dup_rows.nonzero()[0, 0]].tolist()}"
    )


def _scores(logits, bias):
    return logits.float() + bias.float()


def _expected_order(scores, topk):
    """The kernel's slot order: score descending, ties to the lowest index."""
    # A stable sort on the negated scores keeps index order among equals.
    return torch.sort(-scores, dim=-1, stable=True).indices[:, :topk].int()


def _assert_bit_identical(weights, ids, ref_weights, ref_ids, renormalize, scores):
    """Same experts, in the same slot order, with bitwise equal weights.

    Rows whose selected scores contain an exact tie are compared as a set
    (torch.topk orders tied experts arbitrarily); their slot order is checked
    against the kernel's own rule instead.
    """
    selected = scores.gather(1, ref_ids.long())
    tied = (selected[:, 1:] == selected[:, :-1]).any(dim=-1)

    bad_ids = (ids != ref_ids).any(dim=-1) & ~tied
    assert not bad_ids.any(), (
        f"expert selection differs in {int(bad_ids.sum())}/{ids.shape[0]} rows, "
        f"e.g. row {int(bad_ids.nonzero()[0, 0])}: fused "
        f"{ids[bad_ids][0].tolist()} vs torch {ref_ids[bad_ids][0].tolist()}"
    )
    assert torch.equal(ids[tied], _expected_order(scores[tied], ids.shape[1]))

    # Compare weights per expert, so tied rows line up too.
    order, ref_order = ids.argsort(dim=-1), ref_ids.argsort(dim=-1)
    assert torch.equal(
        ids.gather(1, order), ref_ids.gather(1, ref_order)
    ), "selected expert sets differ"
    weights, ref_weights = weights.gather(1, order), ref_weights.gather(1, ref_order)
    if renormalize:
        # The weight sum is reduced in a different order than inductor's;
        # allow a few ulp.
        torch.testing.assert_close(weights, ref_weights, rtol=5e-7, atol=1e-30)
    else:
        bad_w = weights.view(torch.int32) != ref_weights.view(torch.int32)
        assert not bad_w.any(), (
            f"{int(bad_w.sum())} weights differ bitwise, max abs diff "
            f"{(weights - ref_weights).abs().max().item():.3e}"
        )


def _check_against_reference(logits, bias, topk, renormalize):
    ref_weights, ref_ids = _reference(logits, bias, topk, renormalize)
    weights, ids = _fused(logits, bias, topk, renormalize)
    _assert_well_formed(weights, ids, logits.shape[0], logits.shape[1], topk)
    _assert_bit_identical(
        weights, ids, ref_weights, ref_ids, renormalize, _scores(logits, bias)
    )


# -- Random inputs ------------------------------------------------------------


# 767/768/769 straddle the switch from one token per program to several.
@pytest.mark.parametrize(
    "num_tokens", [1, 2, 7, 64, 255, 256, 767, 768, 769, 4096, 32768]
)
@pytest.mark.parametrize(
    "distribution", ["randn", "bias_dominated", "zero_bias", "saturated"]
)
def test_real_config_matches_reference(num_tokens, distribution):
    """Kolibri 1's 384 experts / top-6 across decode and prefill batch sizes."""
    logits, bias = _make_inputs(num_tokens, REAL_EXPERTS, distribution)
    _check_against_reference(logits, bias, REAL_TOP_K, renormalize=False)


@pytest.mark.parametrize(
    "num_experts,topk",
    [
        (8, 2),  # the tiny test checkpoint
        (384, 1),
        (384, 8),  # topk already a power of two
        (384, 16),
        (100, 7),  # neither a power of two
        (129, 3),  # just past a power of two: BLOCK_E=256, 127 padded columns
        (256, 8),  # exact power of two: no padded columns
        (512, 22),  # Nemotron-sized
        (1024, 8),  # BLOCK_T=1 for prefill too
        (2048, 8),  # BLOCK_T=1 even for prefill
        (6, 6),  # topk == num_experts
    ],
)
@pytest.mark.parametrize("num_tokens", [1, 1025, 4096])
def test_shapes_match_reference(num_experts, topk, num_tokens):
    logits, bias = _make_inputs(num_tokens, num_experts, "randn", seed=num_experts)
    _check_against_reference(logits, bias, topk, renormalize=False)


@pytest.mark.parametrize(
    "num_experts,topk,num_tokens",
    [(384, 6, 1), (384, 6, 4096), (100, 7, 1025), (6, 6, 1025)],
)
def test_renormalize_matches_reference(num_experts, topk, num_tokens):
    """Kolibri 1 runs with norm_topk_prob=False; the renormalize branch only
    needs a few shapes, including topk == num_experts."""
    logits, bias = _make_inputs(num_tokens, num_experts, "randn", seed=num_experts)
    _check_against_reference(logits, bias, topk, renormalize=True)


# -- Input variants -----------------------------------------------------------


@pytest.mark.parametrize("logits_dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("bias_dtype", [torch.float32, torch.bfloat16])
def test_input_dtypes(logits_dtype, bias_dtype):
    """Kolibri 1 feeds float32 logits and a float32 bias, but the hook accepts
    any router_logits_dtype; both sides must upcast the same way."""
    logits, bias = _make_inputs(1025, REAL_EXPERTS, "randn")
    # Low-precision inputs produce exact score ties, handled as sets.
    logits, bias = logits.to(logits_dtype), bias.to(bias_dtype)
    _check_against_reference(logits, bias, REAL_TOP_K, renormalize=False)


def test_row_strided_input():
    """A row-strided view (e.g. a slice of a wider router output) is read via
    stride(0) without a copy."""
    wide, bias = _make_inputs(1025, 512, "randn")
    logits, bias = wide[:, :REAL_EXPERTS], bias[:REAL_EXPERTS]
    assert logits.stride(0) == 512 and not logits.is_contiguous()
    _check_against_reference(logits, bias, REAL_TOP_K, renormalize=False)


@pytest.mark.parametrize(
    "malform",
    [
        lambda l, b, k: (l.t().contiguous().t(), b, k),  # column-strided logits
        lambda l, b, k: (l, b, REAL_EXPERTS + 1),  # topk > num_experts
        lambda l, b, k: (l, torch.cat([b, b])[::2], k),  # strided bias
        lambda l, b, k: (l, b[:-1].contiguous(), k),  # bias too short
        lambda l, b, k: (l, torch.cat([b, b[:1]]), k),  # bias too long
        lambda l, b, k: (l, b[None, :], k),  # 2-D bias
        lambda l, b, k: (l, b.cpu(), k),  # bias on another device
    ],
    ids=[
        "logits_col_stride",
        "topk",
        "bias_stride",
        "bias_short",
        "bias_long",
        "bias_2d",
        "bias_device",
    ],
)
def test_input_contract_is_enforced(malform):
    """The kernel reads logits rows and the bias with unit stride and never
    checks sizes, so a malformed input would silently select wrong experts."""
    logits, bias = _make_inputs(64, REAL_EXPERTS, "randn")
    with pytest.raises(AssertionError):
        _fused(*malform(logits, bias, REAL_TOP_K), False)


def test_empty_batch():
    logits = torch.empty(0, REAL_EXPERTS, device="cuda")
    bias = torch.zeros(REAL_EXPERTS, device="cuda")
    weights, ids = _fused(logits, bias, REAL_TOP_K, False)
    _assert_well_formed(weights, ids, 0, REAL_EXPERTS, REAL_TOP_K)


# -- Ties and non-finite logits -----------------------------------------------


def _tied_inputs(num_tokens, num_winners, num_tied, seed=0):
    """Each row: ``num_winners`` distinct high scores, then ``num_tied``
    experts sharing one score, the rest far below. Expert positions are
    random per row."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.full((num_tokens, REAL_EXPERTS), -5.0, device="cuda")
    for row in range(num_tokens):
        perm = torch.randperm(REAL_EXPERTS, device="cuda", generator=g)
        winners = perm[:num_winners]
        logits[row, winners] = (
            torch.arange(num_winners, 0, -1, device="cuda", dtype=torch.float32) + 1.0
        )
        logits[row, perm[num_winners : num_winners + num_tied]] = 0.5
    return logits, torch.zeros(REAL_EXPERTS, device="cuda")


@pytest.mark.parametrize(
    "num_winners,num_tied",
    [
        (0, 20),  # all top-k slots tied, tie crosses the boundary
        (3, 3),  # tie fills the top-k exactly
        (2, 2),  # tie inside the top-k
        (5, 20),  # tie decides only the last slot
    ],
)
def test_ties_select_same_set_as_torch(num_winners, num_tied):
    logits, bias = _tied_inputs(256, num_winners, num_tied)
    _check_against_reference(logits, bias, REAL_TOP_K, renormalize=False)
    # A tie at the top-k boundary goes to the lowest expert index, as in
    # torch.topk on CUDA; the check above also compares the selected sets.
    _, ids = _fused(logits, bias, REAL_TOP_K, False)
    assert torch.equal(ids, _expected_order(_scores(logits, bias), REAL_TOP_K))


def test_all_scores_equal():
    logits = torch.zeros(4, REAL_EXPERTS, device="cuda")
    bias = torch.zeros(REAL_EXPERTS, device="cuda")
    weights, ids = _fused(logits, bias, REAL_TOP_K, False)
    assert ids.tolist() == [list(range(REAL_TOP_K))] * 4
    assert torch.equal(weights, torch.full_like(weights, 0.5))


def test_neg_inf_logits_give_distinct_experts():
    """A row with fewer finite scores than top-k must still route to top-k
    distinct experts (torch.topk does); duplicate ids would count one expert
    several times."""
    logits = torch.full((2, REAL_EXPERTS), float("-inf"), device="cuda")
    logits[0, :3] = torch.tensor([1.0, 2.0, 3.0], device="cuda")
    bias = torch.zeros(REAL_EXPERTS, device="cuda")
    weights, ids = _fused(logits, bias, REAL_TOP_K, False)
    _assert_well_formed(weights, ids, 2, REAL_EXPERTS, REAL_TOP_K)
    assert ids[0, :3].tolist() == [2, 1, 0]
    assert torch.equal(weights[:, 3:], torch.zeros_like(weights[:, 3:]))
    # The -inf experts tie, so torch agrees as a set.
    _check_against_reference(logits, bias, REAL_TOP_K, renormalize=False)


def test_pos_inf_logits():
    logits, bias = _make_inputs(4, REAL_EXPERTS, "randn")
    logits[:, [7, 9]] = float("inf")
    # The two +inf experts tie for the first two slots.
    _check_against_reference(logits, bias, REAL_TOP_K, renormalize=False)


def test_nan_logits_stay_in_range():
    """NaN router logits mean the forward pass is already broken, but the ids
    still index expert weights: they must stay valid and distinct. (torch.topk
    ranks NaN highest; the kernel skips it, so the two are not compared.)"""
    logits, bias = _make_inputs(4, REAL_EXPERTS, "randn")
    logits[0, 5] = float("nan")
    logits[1, :] = float("nan")
    weights, ids = _fused(logits, bias, REAL_TOP_K, False)
    _assert_well_formed(weights, ids, 4, REAL_EXPERTS, REAL_TOP_K)
    _check_against_reference(logits[2:], bias, REAL_TOP_K, renormalize=False)


# -- Execution properties -----------------------------------------------------


@pytest.mark.parametrize("num_tokens", [1, 8, 256])
def test_cuda_graph_replay(num_tokens):
    """Decode runs the router inside a captured CUDA graph: replaying it on
    new logits must give the eager result."""
    logits, bias = _make_inputs(num_tokens, REAL_EXPERTS, "randn")
    _fused(logits, bias, REAL_TOP_K, False)  # compile outside the capture

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        weights, ids = _fused(logits, bias, REAL_TOP_K, False)

    for seed in range(3):
        new_logits, new_bias = _make_inputs(num_tokens, REAL_EXPERTS, "randn", seed + 1)
        logits.copy_(new_logits)
        bias.copy_(new_bias)
        graph.replay()
        ref_weights, ref_ids = _reference(logits, bias, REAL_TOP_K, False)
        _assert_bit_identical(
            weights, ids, ref_weights, ref_ids, False, _scores(logits, bias)
        )


# -- Model hook ---------------------------------------------------------------


def test_kolibri1_routing_dispatch():
    """``kolibri1_routing`` (the FusedMoE custom_routing_function) uses the
    fused kernel on CUDA and the torch reference on CPU; both agree."""
    from aleph_alpha_inference.kolibri1 import kolibri1_routing

    logits, bias = _make_inputs(257, REAL_EXPERTS, "bias_dominated")
    hidden = torch.zeros(257, 16, device="cuda")

    weights, ids = kolibri1_routing(hidden, logits, REAL_TOP_K, False, bias)
    ref_weights, ref_ids = _reference(logits, bias, REAL_TOP_K, False)
    _assert_bit_identical(
        weights, ids, ref_weights, ref_ids, False, _scores(logits, bias)
    )

    cpu_weights, cpu_ids = kolibri1_routing(
        hidden.cpu(), logits.cpu(), REAL_TOP_K, False, bias.cpu()
    )
    assert cpu_ids.device.type == "cpu"
    assert torch.equal(cpu_ids, ids.cpu())
    torch.testing.assert_close(cpu_weights, weights.cpu(), rtol=1e-6, atol=0)

    # Column-strided logits are made contiguous before the kernel.
    t_weights, t_ids = kolibri1_routing(
        hidden, logits.t().contiguous().t(), REAL_TOP_K, False, bias
    )
    assert torch.equal(t_ids, ids) and torch.equal(t_weights, weights)
