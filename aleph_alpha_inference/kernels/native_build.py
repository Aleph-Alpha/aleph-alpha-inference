# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""JIT build of the packaged C++/CUDA extensions."""

import hashlib
import json
import logging
import os
from pathlib import Path

# Under `vllm.` so vLLM's logging config shows it.
logger = logging.getLogger(f"vllm.{__name__}")


def load_extension(name, sources, **options):
    """Build (or reuse) and import the torch extension compiled from `sources`.

    `options` are passed to `torch.utils.cpp_extension.load` unchanged. The
    module name is `aleph_alpha_inference_<name>_<hash>`, the hash covering the
    source bytes, `options` and the torch version, so plugin and torch versions
    sharing `TORCH_EXTENSIONS_DIR` never load or rebuild each other's modules.
    """
    import torch
    from torch.utils import cpp_extension

    from aleph_alpha_inference.cuda_toolkit import prepare_cuda_toolkit

    # Also done when vLLM loads the plugin; here for builds outside vLLM.
    prepare_cuda_toolkit()
    if cpp_extension.CUDA_HOME is None and (home := os.environ.get("CUDA_HOME")):
        # torch reads CUDA_HOME once, when `cpp_extension` is first imported,
        # and vLLM imports it before it loads plugins.
        cpp_extension.CUDA_HOME = home

    sources = [str(source) for source in sources]
    digest = hashlib.sha256()
    for source in sources:
        digest.update(Path(source).read_bytes())
    digest.update(json.dumps(options, sort_keys=True, default=str).encode())
    digest.update(torch.__version__.encode())
    module = f"aleph_alpha_inference_{name}_{digest.hexdigest()[:12]}"
    # The first load in a fresh cache compiles for up to a few minutes.
    logger.info("Loading native extension %s (compiles on first use)", module)
    return cpp_extension.load(name=module, sources=sources, verbose=False, **options)
