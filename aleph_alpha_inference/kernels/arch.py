# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""Which serving kernels the local GPU runs."""

from functools import lru_cache

import torch


@lru_cache(None)
def is_blackwell():
    """Datacenter Blackwell (sm_100/sm_103): hardware E2M1 conversion and
    tcgen05 block-scaled FP4 scans. Hopper runs the mma.sync scans with the
    software conversion instead. Without a CUDA device (CPU test lanes) the
    Blackwell paths are assumed."""
    return not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] == 10


@lru_cache(None)
def gencode():
    """The nvcc target of the local device for the CUDA C++ extensions."""
    if is_blackwell():
        return "-gencode=arch=compute_100f,code=sm_100f"
    major, minor = torch.cuda.get_device_capability()
    return f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"
