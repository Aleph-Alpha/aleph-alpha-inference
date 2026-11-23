# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""Shape-agnostic CuTe executables launched through TVM-FFI, as FA4 does.

An executable compiles once per semantic key against fake tensors whose row
count is symbolic, so one executable serves every batch, request count and
KV-cache size. The trailing dims and strides stay static: the kernels tile and
vectorize on them, and they are fixed per call site. Launches pass torch
tensors and Python scalars straight to TVM-FFI on the current stream.

With FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 (FA4's switch), executables also
persist in FA4's JITPersistentCache under its cache directory, namespaced by a
fingerprint of these sources, so restarts load instead of compiling.
"""

import hashlib
import sys
from functools import lru_cache
from pathlib import Path

import cutlass
import cutlass.cute as cute
import torch

try:
    # Serving shares vLLM's FA4 cache machinery and its switches.
    from vllm.vllm_flash_attn.cute import cache_utils
except ImportError:
    from flash_attn.cute import cache_utils

_DTYPES = {
    torch.float32: cutlass.Float32,
    torch.bfloat16: cutlass.BFloat16,
    torch.float16: cutlass.Float16,
    torch.int64: cutlass.Int64,
    torch.int32: cutlass.Int32,
    torch.int8: cutlass.Int8,
    torch.uint8: cutlass.Uint8,
    torch.float8_e4m3fn: cutlass.Float8E4M3FN,
    torch.float8_e5m2: cutlass.Float8E5M2,
}


@lru_cache(maxsize=1)
def fingerprint():
    """Hash of the Kolibri kernel sources, FA4's sources and the DSL versions."""
    h = hashlib.sha256(cache_utils._compute_source_fingerprint().encode())
    for root in (
        Path(__file__).parent,
        Path(__file__).parent.parent / "model",
    ):
        for source in sorted(root.glob("*.py")):
            h.update(source.name.encode())
            h.update(source.read_bytes())
    return h.hexdigest()


# Every Kolibri executable cache, by name, for audits.
CACHES = {}


def jit_cache(name):
    """An executable cache, persistent when FA4's persistent cache is enabled."""
    if name in CACHES:
        raise ValueError(f"Duplicate CuTe executable cache {name}")
    if cache_utils.CUTE_DSL_CACHE_ENABLED:
        path = cache_utils.get_cache_path() / "kolibri" / fingerprint()[:32] / name
        CACHES[name] = cache_utils.JITPersistentCache(path)
    else:
        CACHES[name] = cache_utils.JITCache()
    return CACHES[name]


def tensor_spec(t, align, dynamic=False):
    """The static part of `t`'s signature: dtype, trailing dims, strides and
    alignment; only dtype, rank and alignment for `dynamic` tensors."""
    if dynamic:
        return str(t.dtype), t.ndim, align
    return str(t.dtype), tuple(t.shape[1:]), tuple(t.stride()), align


def fake_tensor(spec):
    """A fake tensor with a symbolic row count; a `dynamic` spec has symbolic
    dims and strides except its unit-stride last dim."""
    dtype, align = _DTYPES[getattr(torch, spec[0].removeprefix("torch."))], spec[-1]
    if len(spec) == 3:
        rank = spec[1]
        shape = tuple(cute.sym_int() for _ in range(rank))
        stride = (*(cute.sym_int64() for _ in range(rank - 1)), 1)
    else:
        shape, stride = (cute.sym_int(), *spec[1]), spec[2]
    return cute.runtime.make_fake_tensor(
        dtype, shape, stride=stride, assumed_align=align
    )


def compile_tvm_ffi(op, *args, options=""):
    """Compile `op` for `args` (fake tensors and scalar placeholders) with the
    current TVM-FFI environment stream as its final argument."""
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        op, *args, stream, options=f"--enable-tvm-ffi {options}".strip()
    )


class Executables:
    """Executables of one launcher, keyed by semantic config and tensor specs.

    Keys never hold sizes that vary at run time, so the count stays bounded by
    the call sites' configurations; nothing is evicted (graphs replay them).
    """

    def __init__(self, name):
        self.cache = jit_cache(name)

    def __len__(self):
        return len(self.cache.cache)

    def __iter__(self):
        return iter(self.cache.cache)

    def clear(self):
        """Drop the in-memory executables; persistent ones stay on disk."""
        self.cache.cache.clear()

    def get(self, key, build):
        if key not in self.cache:
            self.cache[key] = build()
        return self.cache[key]

    def launch(
        self, op, tensors, *, key=(), scalars=(), aligns=None, dynamic=(), options=""
    ):
        """Run `op(*tensors, *scalars, stream)`, compiling it once per key.
        `scalars` are Python ints, passed as Int32 at run time; tensors at the
        `dynamic` positions have fully dynamic layouts."""
        if aligns is None:
            aligns = [16 if t.dtype == torch.uint8 else 4 for t in tensors]
        specs = tuple(
            tensor_spec(t, a, i in dynamic)
            for i, (t, a) in enumerate(zip(tensors, aligns))
        )
        signature = type(op).__qualname__, key, specs, len(scalars)

        def build():
            placeholders = [cutlass.Int32(0)] * len(scalars)
            return compile_tvm_ffi(
                op, *map(fake_tensor, specs), *placeholders, options=options
            )

        tensors = [t.detach() if t.requires_grad else t for t in tensors]
        self.get(signature, build)(*tensors, *scalars)


def executable_counts():
    """Executables held per Kolibri cache, and in FA4's forward caches."""
    counts = {name: len(cache.cache) for name, cache in CACHES.items()}
    for module, attr in (
        ("flash_attn.cute.interface", "_flash_attn_fwd"),
        ("vllm.vllm_flash_attn.cute.interface", "_flash_attn_fwd"),
    ):
        fn = getattr(sys.modules.get(module), attr, None)
        if fn is not None:
            counts[f"{module}.{attr}"] = len(fn.compile_cache.cache)
    return counts
