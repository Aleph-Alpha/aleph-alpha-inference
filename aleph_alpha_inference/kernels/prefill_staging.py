# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""Shared scratch for sequential staged prefill.

BF16 cache staging uses one 2 GiB workspace per worker, across all requests
and attention layers. It is a temporary prefill buffer, not an additional
persistent KV cache. Overlapping ubatches disable staging.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch

from aleph_alpha_inference.kernels.arch import is_blackwell
from aleph_alpha_inference.kernels.cute_jit import Executables
from aleph_alpha_inference.kernels.fp16_fragments import (
    dequant_word,
    f16x2_to_bf16x2,
    scale_pair,
)

# Tokens per cache page; the consumers of the staged context require it too.
PAGE = 128
MAX_TOKENS = 1048576
_WORKSPACES = {}
# What each device's workspace holds: (step, storage address, request,
# tokens) of the last staging, or nothing if unknown. Overlapping ubatches
# disable staging, so one record per device suffices.
_STAGED = {}
_COMPILED = Executables("prefill_staging")


def workspace(device):
    """The staging buffer of `device`, allocated on first use.

    A [2, MAX_TOKENS, 1, 4, 128] BF16 tensor, K first and V second. Every
    request and attention layer of this process shares it.
    """
    device = torch.device(device)
    if device not in _WORKSPACES:
        _WORKSPACES[device] = torch.empty(
            2, MAX_TOKENS, 1, 4, 128, device=device, dtype=torch.bfloat16
        )
    return _WORKSPACES[device]


class DequantizeContext:
    """Launcher of the kernel that writes one request's NVFP4 K/V as BF16.

    `storage` holds NVFP4 pages of `PAGE` tokens, `table[request]` the
    request's pages and `lengths[request]` its token count. `output` is
    [2, capacity, 1, 4, 128] BF16. The kernel writes rows below
    `min(tokens, lengths[request])` and leaves the others as they are.
    """

    @cute.jit
    def __call__(
        self,
        storage: cute.Tensor,
        table: cute.Tensor,
        lengths: cute.Tensor,
        output: cute.Tensor,
        request: cutlass.Int32,
        tokens: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(storage, table, lengths, output, request, tokens).launch(
            grid=(cute.ceil_div(tokens, 8), 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        storage: cute.Tensor,
        table: cute.Tensor,
        lengths: cute.Tensor,
        output: cute.Tensor,
        request: cutlass.Int32,
        tokens: cutlass.Int32,
    ):
        block, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        row = block * 8 + tid // 32
        head = (tid % 32) // 8
        dim = (tid % 8) * 16
        if row < tokens and row < lengths[request]:
            physical = table[request, row // PAGE]
            offset = cutlass.Int64(physical) * storage.stride[0]
            token = row % PAGE
            for kind in cutlass.range_constexpr(2):
                width = 256
                pointer = (
                    storage.iterator
                    + offset
                    + kind * PAGE * width
                    + token * width
                    + head * (width // 4)
                    + dim // 2
                )
                scale_pointer = (
                    storage.iterator
                    + offset
                    + (512 + kind * 32) * PAGE
                    + token * 32
                    + head * 8
                    + dim // 16
                )
                target = (output, kind, row, head, dim)
                if cutlass.const_expr(is_blackwell()):
                    self.dequantize_typed(pointer, scale_pointer, target)
                else:
                    self.dequantize_fragments(pointer, scale_pointer, target)

    @cute.jit
    def output_pointer(self, output, kind, row, head, dim):
        """Address of the 16 BF16 outputs of (`kind`, `row`, `head`, `dim`)."""
        return (
            output.iterator
            + kind * output.stride[0]
            + row * output.stride[1]
            + head * 128
            + dim
        )

    @cute.jit
    def dequantize_typed(self, pointer, scale_pointer, target):
        """Write 16 BF16 values for 16 E2M1 values and their E4M3 scale, with
        the hardware E2M1 conversion of Blackwell. `target` is the argument
        tuple of `output_pointer`."""
        source = cute.make_tensor(
            cute.make_ptr(
                cutlass.Float4E2M1FN,
                pointer.toint(),
                cute.AddressSpace.gmem,
                assumed_align=8,
            ),
            cute.make_layout((16,)),
        )
        scale = cute.make_tensor(
            cute.make_ptr(
                cutlass.Float8E4M3FN,
                scale_pointer.toint(),
                cute.AddressSpace.gmem,
                assumed_align=1,
            ),
            cute.make_layout((1,)),
        )[0].to(cutlass.Float32)
        pointer_out = self.output_pointer(*target)
        dest = cute.make_tensor(
            cute.make_ptr(
                cutlass.BFloat16,
                pointer_out.toint(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            cute.make_layout((16,)),
        )
        # E2M1 * E4M3 is exact in FP32.
        values = source.load().to(cutlass.Float32) * scale
        dest.store(values.to(cutlass.BFloat16))

    @cute.jit
    def dequantize_fragments(self, pointer, scale_pointer, target):
        """`dequantize_typed` for GPUs without the E2M1 conversion (Hopper):
        the software conversion of `fp16_fragments`, bit for bit the same."""
        words = cute.make_tensor(
            cute.make_ptr(
                cutlass.Uint32, pointer.toint(), cute.AddressSpace.gmem, assumed_align=8
            ),
            cute.make_layout((2,)),
        )
        scale_byte = cute.make_tensor(
            cute.make_ptr(
                cutlass.Uint8,
                scale_pointer.toint(),
                cute.AddressSpace.gmem,
                assumed_align=1,
            ),
            cute.make_layout((1,)),
        )[0]
        # The scale in the low half of an FP16 pair; dequant_word broadcasts it.
        scale = scale_pair(cutlass.Uint32(scale_byte))
        pairs = cute.make_rmem_tensor((8,), cutlass.Uint32)
        for w in cutlass.range_constexpr(2):
            # E2M1 * E4M3 is exact in FP16 and in BF16: at most 6 significant
            # bits, within both ranges.
            halves = dequant_word(words[w], scale, 0)
            for i in cutlass.range_constexpr(4):
                pairs[4 * w + i] = f16x2_to_bf16x2(halves[i])
        pointer_out = self.output_pointer(*target)
        dest = cute.make_tensor(
            cute.make_ptr(
                cutlass.Uint32,
                pointer_out.toint(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            cute.make_layout((8,)),
        )
        cute.autovec_copy(pairs, dest)


def stage_context(cache, table, lengths, request, tokens, step=None):
    """Dequantize the first `tokens` positions of one request for prefill.

    `cache` is an NVFP4 cache of 128-token pages, `table` ([requests, pages]
    int32) maps each request's pages and `lengths` ([requests] int32) holds
    the requests' token counts. Returns K and V as [tokens, 1, 4, 128] BF16
    views of `workspace(cache.storage.device)`; the next call overwrites them.
    Rows at or past `lengths[request]` are not written and hold stale data.

    `step` identifies one engine step, for example an object that all layers
    of the step share. A call with the same `step` object, cache, request and
    `tokens` as the last staging on the device returns the staged views
    without staging again. Callers must not write `cache` within a step after
    staging it from that cache. A call without `step` always stages.

    Pre-condition that is not checked: the data of `cache.storage` starts at
    an 8-byte aligned address and its page stride is a multiple of 8 bytes.
    """
    if not 0 < tokens <= MAX_TOKENS:
        raise ValueError(
            f"Prefill staging holds 1 to {MAX_TOKENS} tokens, got {tokens}"
        )
    device = cache.storage.device
    output = workspace(device)
    key = (cache.storage.data_ptr(), request, tokens)
    staged = _STAGED.get(device)
    if step is not None and staged is not None:
        if staged[0] is step and staged[1:] == key:
            return output[0, :tokens], output[1, :tokens]
    # Record nothing while staging, so a failed launch leaves no stale record.
    _STAGED.pop(device, None)
    _COMPILED.launch(
        DequantizeContext(),
        (cache.storage, table, lengths, output),
        scalars=(request, tokens),
        # What the kernel assumes of the cache loads and the output stores.
        aligns=(8, 4, 4, 16),
    )
    if step is not None:
        _STAGED[device] = (step, *key)
    return output[0, :tokens], output[1, :tokens]
