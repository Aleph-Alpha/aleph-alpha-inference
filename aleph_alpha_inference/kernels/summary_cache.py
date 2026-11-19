# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""Paged NVFP4 KV: packed E2M1 values with one E4M3 scale per 16 values,
NVFP4 routing keys, and allocator-accounted summaries and partial sums."""

from functools import lru_cache
from pathlib import Path

import torch

from aleph_alpha_inference.kernels.arch import gencode
from aleph_alpha_inference.kernels.native_build import load_extension


@lru_cache(None)
def native():
    """The summary append extension, built on first use."""
    return load_extension(
        "nvfp4_summary_cache",
        [Path(__file__).with_suffix(".cu")],
        extra_cuda_cflags=["-O3", gencode()],
    )


def page_rows(tensor, rows, width):
    """Validate a [pages, bytes] uint8 tensor whose rows hold `width` bytes."""
    if (
        tensor.ndim != 2
        or tensor.dtype != torch.uint8
        or tensor.stride(1) != 1
        or tensor.size(1) < width
        or (rows is not None and tensor.size(0) != rows)
    ):
        raise ValueError("NVFP4 cache requires byte storage containing complete pages")
    return tensor


class SummaryCache:
    """Paged NVFP4 K/V with routing keys and block summaries.

    Every tensor is [pages, bytes] uint8, indexed by the same page ids. Their
    rows may be strided views into one allocation, as vLLM packs pages.

    Args:
        storage: K and V of every token and their scales.
        routing: The NVFP4 keys of KV head 0 and their scales.
        summaries: Per `BLOCK` tokens, the NVFP4 mean of KV head 0's keys
            (64 bytes each), then the means' scales (8 bytes each).
        partial: The FP32 key sum of the page's open block.
        page: Tokens per page.

    Raises:
        ValueError: If `page` is not 128, a tensor does not hold complete
            pages or the tensors are on different devices.
    """

    # Per token: K and V (256 bytes each), then their scales (32 bytes each).
    STORAGE_BYTES_PER_TOKEN = 576
    # Per token: 64 routing key bytes, then 8 scale bytes, region-major per page.
    ROUTING_BYTES_PER_TOKEN = 72
    BLOCK = 16
    SUMMARY_BYTES = 72
    PARTIAL_BYTES = 512

    def __init__(self, storage, routing, summaries, partial, page=128):
        if page != 128:
            raise ValueError("NVFP4 cache requires 128-token pages")
        self.page = page
        self.storage = page_rows(storage, None, page * self.STORAGE_BYTES_PER_TOKEN)
        pages = storage.shape[0]
        self.routing = page_rows(routing, pages, page * self.ROUTING_BYTES_PER_TOKEN)
        blocks = page // self.BLOCK
        packed_bytes = blocks * 64
        page_rows(summaries, pages, blocks * self.SUMMARY_BYTES)
        page_rows(partial, pages, self.PARTIAL_BYTES)
        self.tensors = (storage, routing, summaries, partial)
        if any(tensor.device != storage.device for tensor in self.tensors):
            raise ValueError("NVFP4 cache tensors must share a device")

        def region(tensor, start, width, dtype, shape):
            return (
                tensor[:, start * page : (start + width) * page]
                .view(dtype)
                .view(pages, page, *shape)
            )

        self.k = region(storage, 0, 256, torch.uint8, (4, 64))
        self.v = region(storage, 256, 256, torch.uint8, (4, 64))
        self.ks = region(storage, 512, 32, torch.float8_e4m3fn, (4, 8))
        self.vs = region(storage, 544, 32, torch.float8_e4m3fn, (4, 8))
        self.rk = region(routing, 0, 64, torch.int8, (64,))
        self.rks = region(routing, 64, 8, torch.int32, (2,))
        self.summaries = summaries[:, :packed_bytes].view(pages, blocks, 64)
        self.summary_scales = (
            summaries[:, packed_bytes : blocks * self.SUMMARY_BYTES]
            .view(torch.int32)
            .view(pages, blocks, 2)
        )
        self.partial = partial[:, : self.PARTIAL_BYTES].view(torch.float32)
        native()  # Build the extension before serving, not on the first append.

    def append(self, k, v, slots):
        """Quantize and store K/V rows, head 0's routing keys and the
        summaries of the blocks they complete.

        Rows that write to the same page must hold consecutive, increasing
        slots. This is not checked.

        Args:
            k: [rows, 4, 128] BF16 keys with 32-byte aligned rows.
            v: The values, shaped and aligned like `k`.
            slots: [rows] int64 token slots (`page * 128 + offset`); negative
                slots are skipped.
        """
        native().append_summary(
            k,
            v,
            slots,
            self.storage,
            self.routing,
            self.summaries,
            self.summary_scales,
            self.partial,
        )

    @classmethod
    def page_bytes(cls, page):
        """Bytes per page of each tensor the constructor takes, in order."""
        return (
            page * cls.STORAGE_BYTES_PER_TOKEN,
            page * cls.ROUTING_BYTES_PER_TOKEN,
            page // cls.BLOCK * cls.SUMMARY_BYTES,
            cls.PARTIAL_BYTES,
        )

    @classmethod
    def allocate(cls, pages, page=128, device="cuda"):
        """A cache of `pages` uninitialised pages, one allocation per tensor."""
        return cls(
            *(
                torch.empty(pages, width, device=device, dtype=torch.uint8)
                for width in cls.page_bytes(page)
            ),
            page=page,
        )
