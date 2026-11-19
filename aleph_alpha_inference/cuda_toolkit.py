# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""Make the CUDA toolkit from pip (the `cuda` extra) usable for JIT builds.

The plugin's CUDA extensions and FlashInfer's kernels compile on first use.
Without a system toolkit, the pip toolkit in `site-packages/nvidia/cu13` has
two gaps for that:

- It is not on `PATH`, so torch and FlashInfer do not find it unless
  `CUDA_HOME` points at it.
- Its libraries have only their versioned names, such as `libcudart.so.13`
  and `libnvrtc.so.13`. The linker needs the unversioned `libcudart.so` for
  the `-lcudart` that torch and FlashInfer link with, and likewise for
  `-lnvrtc`, `-lcublas` and the others.

`prepare_cuda_toolkit` closes both. A system toolkit and an explicit
`CUDA_HOME` take precedence; the pip toolkit is only the last resort.
"""

import hashlib
import importlib.util
import logging
import os
import re
import shutil
from functools import cache
from pathlib import Path

# A child of vLLM's logger, without importing vLLM.
logger = logging.getLogger(f"vllm.{__name__}")

# Where torch and FlashInfer look when neither `CUDA_HOME` nor `nvcc` is set.
DEFAULT_CUDA_HOME = Path("/usr/local/cuda")

# `libnvrtc.so.13` and `libnvrtc.so.13.0.96` are versions of `libnvrtc.so`.
VERSIONED_LIBRARY = re.compile(r"(lib.+\.so)\.\d[\d.]*")


def pip_cuda_home() -> Path | None:
    """The root of the CUDA toolkit installed from pip, if there is one."""
    spec = importlib.util.find_spec("nvidia")
    if spec is None or not spec.submodule_search_locations:
        return None
    for location in spec.submodule_search_locations:
        # The newest CUDA major version first, e.g. cu13 before cu12.
        for nvcc in sorted(Path(location).glob("cu*/bin/nvcc"), reverse=True):
            return nvcc.parent.parent
    return None


def configured_cuda_home() -> Path | None:
    """The toolkit torch and FlashInfer find on their own: `CUDA_HOME` or
    `CUDA_PATH`, else the one `nvcc` on `PATH` belongs to, else
    `/usr/local/cuda`."""
    if home := os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH"):
        return Path(home)
    if nvcc := shutil.which("nvcc"):
        return Path(nvcc).parent.parent
    if DEFAULT_CUDA_HOME.exists():
        return DEFAULT_CUDA_HOME
    return None


def cache_root() -> Path:
    """Where the plugin keeps its library links."""
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "aleph_alpha_inference" / "cuda_link_dirs"


def unlinkable_libraries(home: Path) -> dict[str, Path]:
    """The libraries of the toolkit in `home` that have only versioned names,
    by their unversioned name, each with its shortest versioned name: the
    soname `libnvrtc.so.13` before `libnvrtc.so.13.0.96`."""
    lib_dirs = [
        lib_dir for lib_dir in (home / "lib64", home / "lib") if lib_dir.is_dir()
    ]
    names = {path.name for lib_dir in lib_dirs for path in lib_dir.iterdir()}
    libraries: dict[str, Path] = {}
    for lib_dir in lib_dirs:
        for path in sorted(lib_dir.iterdir(), key=lambda path: len(path.name)):
            match = VERSIONED_LIBRARY.fullmatch(path.name)
            if match and match[1] not in names:
                libraries.setdefault(match[1], path)
    return libraries


def link_dir(home: Path) -> Path | None:
    """A directory with an unversioned link to every library of the toolkit
    in `home` that has only versioned names. None if there is none."""
    libraries = unlinkable_libraries(home)
    if not libraries:
        return None
    # One directory per toolkit, so toolkits in several venvs do not clash.
    digest = hashlib.sha256(str(home.resolve()).encode()).hexdigest()[:12]
    directory = cache_root() / digest
    directory.mkdir(parents=True, exist_ok=True)
    for name, path in libraries.items():
        target = path.resolve()
        link = directory / name
        if link.is_symlink() and link.resolve() == target:
            continue
        # Every vLLM process gets here; replacing the link atomically keeps
        # concurrent ones from seeing it half made.
        staging = directory / f".{name}.{os.getpid()}"
        staging.unlink(missing_ok=True)
        staging.symlink_to(target)
        os.replace(staging, link)
    return directory


@cache
def prepare_cuda_toolkit() -> None:
    """Point `CUDA_HOME` at the pip toolkit if no other toolkit is found, and
    put the unversioned library names the toolkit lacks on the linker's
    `LIBRARY_PATH`.

    The environment is inherited by the build subprocesses and by the
    processes vLLM starts. Failing here must not take the server down, as
    most models never build anything: a problem is logged and builds go
    ahead as they would have.
    """
    try:
        home = configured_cuda_home()
        if home is None:
            home = pip_cuda_home()
            if home is None:
                return
            os.environ["CUDA_HOME"] = str(home)
            logger.info("Using the CUDA toolkit from pip for JIT builds: %s", home)
        directory = link_dir(home)
        if directory is None:
            return
        paths = os.environ.get("LIBRARY_PATH", "").split(os.pathsep)
        if str(directory) not in paths:
            os.environ["LIBRARY_PATH"] = os.pathsep.join(
                [str(directory), *filter(None, paths)]
            )
        logger.info(
            "The CUDA toolkit in %s has only versioned library names; linking "
            "JIT builds against the unversioned names in %s",
            home,
            directory,
        )
    # OSError from the file system, ValueError from `find_spec`.
    except (OSError, ValueError) as e:
        logger.warning("Could not prepare the CUDA toolkit for JIT builds: %s", e)
