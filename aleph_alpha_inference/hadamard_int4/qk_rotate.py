# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""QK-norm with the gains folded into the rotation, as one kernel over the QKV output.

The int4 layers replace the stock ``q_norm``/``k_norm`` pair with a single fused step.
Letting ``n`` be the scaleless RMS normalize (fp32), the maps from `fold_norm_gains`
turn the QKV projection output into

    q = n(q_raw) @ M_q        with M_q = diag(g_q g_k) R^T
    k = n(k_raw) @ M_k        with M_k = R^T

The 128x128 products run on tensor cores (fp16 operands, fp32 accumulation), which is
more precise than the stock path's bf16 RMSNorm output. The grid is persistent: each
program loads ``M_q`` once and sweeps its q row blocks with it, while the k rows -- a
twelfth of the rows at Kolibri's GQA ratio -- load ``M_k`` per block. Memory-bound
throughout, so this costs about what the stock QK-norm did.
"""

import torch
from vllm.triton_utils import tl, triton

from aleph_alpha_inference.hadamard_int4 import HEAD_DIM


@triton.jit
def _rotate_kernel(
    qkv_ptr,
    q_map_ptr,
    k_map_ptr,
    q_ptr,
    k_ptr,
    tokens,
    stride_qkv,
    stride_q,
    stride_k,
    eps,
    HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    ROWS: tl.constexpr,
):
    """Persistent grid. Each program sweeps q row blocks (token, head) with ``q_map``
    loaded once, then its k row blocks, loading ``k_map`` per block."""
    program = tl.program_id(0)
    programs = tl.num_programs(0)
    dim = tl.arange(0, 128)
    lanes = tl.arange(0, ROWS)
    q_map = tl.load(q_map_ptr + dim[:, None] * 128 + dim[None, :])
    q_total = tokens * HEADS
    q_blocks = tl.cdiv(q_total, ROWS)
    for block in range(program, q_blocks, programs):
        rows = block * ROWS + lanes
        live = rows < q_total
        token = (rows // HEADS).to(tl.int64)
        head = rows % HEADS
        x = tl.load(
            qkv_ptr + token[:, None] * stride_qkv + head[:, None] * 128 + dim[None, :],
            mask=live[:, None],
            other=0.0,
        ).to(tl.float32)
        gain = tl.rsqrt(tl.sum(x * x, 1) / 128 + eps)
        y = tl.dot((x * gain[:, None]).to(tl.float16), q_map)
        tl.store(
            q_ptr + token[:, None] * stride_q + head[:, None] * 128 + dim[None, :],
            y.to(q_ptr.dtype.element_ty),
            mask=live[:, None],
        )
    for block in range(program, tl.cdiv(tokens * KV_HEADS, ROWS), programs):
        rows = block * ROWS + lanes
        live = rows < tokens * KV_HEADS
        token = (rows // KV_HEADS).to(tl.int64)
        head = rows % KV_HEADS
        x = tl.load(
            qkv_ptr
            + token[:, None] * stride_qkv
            + (HEADS + head)[:, None] * 128
            + dim[None, :],
            mask=live[:, None],
            other=0.0,
        ).to(tl.float32)
        gain = tl.rsqrt(tl.sum(x * x, 1) / 128 + eps)
        k_map = tl.load(k_map_ptr + dim[:, None] * 128 + dim[None, :])
        y = tl.dot((x * gain[:, None]).to(tl.float16), k_map)
        tl.store(
            k_ptr + token[:, None] * stride_k + head[:, None] * 128 + dim[None, :],
            y.to(k_ptr.dtype.element_ty),
            mask=live[:, None],
        )


_SMS: dict[torch.device, int] = {}


def rotate_rows(
    qkv: torch.Tensor,
    q_map: torch.Tensor,
    k_map: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    eps: float,
    q: torch.Tensor,
    k: torch.Tensor,
    row_block: int | None = None,
) -> None:
    """qkv ``[T, (num_heads + 2 num_kv_heads) * 128]`` (rows may be strided) -> writes
    q ``[T, num_heads * 128]`` and k ``[T, num_kv_heads * 128]`` in place."""
    tokens = qkv.shape[0]
    if tokens == 0:
        return
    assert qkv.stride(-1) == 1 and q_map.dtype == k_map.dtype == torch.float16
    if row_block is None:
        # small batches want more, smaller blocks so the persistent grid covers the SMs;
        # large ones want fewer sweeps to amortize the q map load
        rows = tokens * num_heads
        row_block = 16 if rows <= 1024 else (32 if rows <= 8192 else 64)
    device = qkv.device
    if device not in _SMS:
        _SMS[device] = torch.cuda.get_device_properties(device).multi_processor_count
    blocks = triton.cdiv(tokens * num_heads, row_block) + triton.cdiv(
        tokens * num_kv_heads, row_block
    )
    _rotate_kernel[(min(blocks, 4 * _SMS[device]),)](
        qkv,
        q_map,
        k_map,
        q,
        k,
        tokens,
        qkv.stride(0),
        q.stride(0),
        k.stride(0),
        eps,
        HEADS=num_heads,
        KV_HEADS=num_kv_heads,
        ROWS=row_block,
        num_warps=4,
    )


@torch.library.custom_op("aleph_alpha_inference::kolibri1_qk_rotate", mutates_args=())
def rotate_qk(
    qkv: torch.Tensor,
    q_map: torch.Tensor,
    k_map: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = qkv.shape[0]
    q = torch.empty(tokens, num_heads * HEAD_DIM, device=qkv.device, dtype=qkv.dtype)
    k = torch.empty(tokens, num_kv_heads * HEAD_DIM, device=qkv.device, dtype=qkv.dtype)
    rotate_rows(qkv, q_map, k_map, num_heads, num_kv_heads, eps, q, k)
    return q, k


@rotate_qk.register_fake
def _(qkv, q_map, k_map, num_heads, num_kv_heads, eps):
    tokens = qkv.shape[0]
    return qkv.new_empty(tokens, num_heads * HEAD_DIM), qkv.new_empty(
        tokens, num_kv_heads * HEAD_DIM
    )


def rotate_qk_reference(
    qkv: torch.Tensor,
    q_gain: torch.Tensor,
    k_gain: torch.Tensor,
    rotation: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 reference: q = R (g_q g_k n(q)), k = R n(k)."""
    tokens = qkv.shape[0]
    x = qkv.float()
    q = x[:, : num_heads * HEAD_DIM].view(tokens, num_heads, HEAD_DIM)
    k = x[:, num_heads * HEAD_DIM : (num_heads + num_kv_heads) * HEAD_DIM].view(
        tokens, num_kv_heads, HEAD_DIM
    )

    def normalize(y):
        return y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)

    rotated = rotation.float().T
    return (
        (normalize(q) * q_gain.float() * k_gain.float()) @ rotated,
        normalize(k) @ rotated,
    )
