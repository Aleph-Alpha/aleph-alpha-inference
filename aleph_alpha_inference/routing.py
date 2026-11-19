# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""Fused sigmoid_logit_add router for Kolibri 1.

torchtitan's ``sigmoid_logit_add`` router selects the top-k experts over
``logits + expert_bias`` and weights them with the unbiased
``sigmoid(logits)``. No FlashInfer/TRT-LLM routing method implements this
(DeepSeek-V3 / MiniMax2 add the bias after the sigmoid and renormalise), so
Kolibri 1 routes through vLLM's ``custom_routing_function`` hook.

Written in torch, that hook costs three kernels per layer: ``torch.topk``'s
gatherTopK + bitonicSort and an inductor gather/sigmoid kernel. At batch 1
they take ~16 us per layer, and at a 32k-token prefill chunk gatherTopK alone
takes ~0.6 ms per layer. The Triton kernel below does bias-add, top-k and the
sigmoid gather in one pass over the router logits.
"""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _sigmoid_logit_add_topk_kernel(
    logits_ptr,
    bias_ptr,
    weights_ptr,
    ids_ptr,
    num_tokens,
    num_experts,
    stride_logits,
    TOPK: tl.constexpr,
    TOPK_PAD: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_E: tl.constexpr,
    RENORMALIZE: tl.constexpr,
    LOWEST_SCORE: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    cols = tl.arange(0, BLOCK_E)
    row_mask = rows < num_tokens
    col_mask = cols < num_experts

    logits = tl.load(
        logits_ptr + rows[:, None] * stride_logits + cols[None, :],
        mask=row_mask[:, None] & col_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    # -inf bias makes padding scores -inf.
    bias = tl.load(bias_ptr + cols, mask=col_mask, other=float("-inf")).to(tl.float32)
    scores = logits + bias[None, :]
    # Raise real experts to at least -FLT_MAX and keep padding at -inf.
    # That way we will always have real experts ranked above padding and
    # already chosen experts.
    floor = tl.where(col_mask, LOWEST_SCORE, float("-inf"))[None, :]
    scores = tl.where(scores > floor, scores, floor)

    slots = tl.arange(0, TOPK_PAD)
    out_w = tl.zeros((BLOCK_T, TOPK_PAD), dtype=tl.float32)
    out_ids = tl.zeros((BLOCK_T, TOPK_PAD), dtype=tl.int32)
    # Iterative argmax, highest score first (matches torch.topk sorted=True).
    # Ties resolve to the lowest expert index. A chosen expert drops to -inf,
    # below every remaining real expert, and topk <= num_experts leaves one,
    # so each row gets TOPK distinct experts.
    for k in tl.static_range(TOPK):
        idx = tl.argmax(scores, axis=1, tie_break_left=True)
        chosen = cols[None, :] == idx[:, None]
        w = tl.sigmoid(tl.sum(tl.where(chosen, logits, 0.0), axis=1))
        out_w = tl.where(slots[None, :] == k, w[:, None], out_w)
        out_ids = tl.where(slots[None, :] == k, idx[:, None], out_ids)
        scores = tl.where(chosen, float("-inf"), scores)

    if RENORMALIZE:
        # torchtitan's route_norm: divide by the weight sum plus 1e-20.
        out_w = out_w / (tl.sum(out_w, axis=1)[:, None] + 1e-20)

    out_offsets = rows[:, None] * TOPK + slots[None, :]
    out_mask = row_mask[:, None] & (slots[None, :] < TOPK)
    tl.store(weights_ptr + out_offsets, out_w, mask=out_mask)
    tl.store(ids_ptr + out_offsets, out_ids, mask=out_mask)


def fused_sigmoid_logit_add_topk(
    gating_output: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (float32 weights, int32 expert ids), each [num_tokens, topk]."""
    assert gating_output.dim() == 2 and gating_output.stride(-1) == 1
    num_tokens, num_experts = gating_output.shape
    assert topk <= num_experts
    # The kernel reads the bias with unit stride and no bounds beyond
    # num_experts.
    assert (
        e_score_correction_bias.dim() == 1
        and e_score_correction_bias.is_contiguous()
        and e_score_correction_bias.numel() == num_experts
        and e_score_correction_bias.device == gating_output.device
    )
    weights = torch.empty(
        num_tokens, topk, dtype=torch.float32, device=gating_output.device
    )
    ids = torch.empty(num_tokens, topk, dtype=torch.int32, device=gating_output.device)
    if num_tokens == 0:
        return weights, ids

    block_e = triton.next_power_of_2(num_experts)
    # One token per program is fastest for small batches; larger ones
    # amortise better with several, up to a 1024-element tile.
    block_t = 1 if num_tokens < 768 else max(1, min(4, 1024 // block_e))
    _sigmoid_logit_add_topk_kernel[(triton.cdiv(num_tokens, block_t),)](
        gating_output,
        e_score_correction_bias,
        weights,
        ids,
        num_tokens,
        num_experts,
        gating_output.stride(0),
        TOPK=topk,
        TOPK_PAD=triton.next_power_of_2(topk),
        BLOCK_T=block_t,
        BLOCK_E=block_e,
        RENORMALIZE=renormalize,
        # Lowest finite float32. Real experts never score below it, which leaves
        # -inf free to mean "cannot be picked".
        LOWEST_SCORE=-torch.finfo(torch.float32).max,
        num_warps=4,
    )
    return weights, ids
