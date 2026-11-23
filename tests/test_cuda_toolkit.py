# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""JIT builds find and link with the CUDA toolkit from pip."""

import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from aleph_alpha_inference import cuda_toolkit
from aleph_alpha_inference.cuda_toolkit import (
    link_dir,
    pip_cuda_home,
    prepare_cuda_toolkit,
)


def fake_toolkit(root: Path, lib_dir: str, *libraries: str) -> Path:
    (root / "bin").mkdir(parents=True)
    (root / "bin" / "nvcc").touch()
    (root / lib_dir).mkdir()
    for library in libraries:
        (root / lib_dir / library).touch()
    return root


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    """No toolkit configured, links cached under `tmp_path`, and
    `prepare_cuda_toolkit` free to run again."""
    for name in ("CUDA_HOME", "CUDA_PATH", "LIBRARY_PATH"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(cuda_toolkit, "DEFAULT_CUDA_HOME", tmp_path / "no-cuda")
    prepare_cuda_toolkit.cache_clear()
    yield
    prepare_cuda_toolkit.cache_clear()


def test_links_the_versioned_runtime(clean_env, tmp_path):
    home = fake_toolkit(
        tmp_path / "cu13", "lib", "libcudart.so.13.0.96", "libcudart.so.13"
    )
    directory = link_dir(home)
    assert directory is not None and directory.is_relative_to(tmp_path / "cache")
    link = directory / "libcudart.so"
    assert link.is_symlink()
    assert link.resolve() == (home / "lib" / "libcudart.so.13").resolve()
    # Again, from the next process: the same link.
    assert link_dir(home) == directory


def test_links_every_versioned_library(clean_env, tmp_path):
    """FlashInfer's MoE also links `-lnvrtc` and `-lcublas`."""
    home = fake_toolkit(
        tmp_path / "cu13",
        "lib",
        "libcudart.so.13",
        "libnvrtc.so.13",
        "libnvrtc-builtins.so.13.0",
        "libcublas.so.13",
        "libcudart_static.a",
        "libnvperf_host.so",
    )
    directory = link_dir(home)
    assert sorted(path.name for path in directory.iterdir()) == [
        "libcublas.so",
        "libcudart.so",
        "libnvrtc-builtins.so",
        "libnvrtc.so",
    ]
    assert (directory / "libnvrtc.so").resolve() == (
        home / "lib" / "libnvrtc.so.13"
    ).resolve()


def test_links_only_what_is_missing(clean_env, tmp_path):
    home = fake_toolkit(
        tmp_path / "cu13", "lib", "libcudart.so", "libcudart.so.13", "libnvrtc.so.13"
    )
    assert [path.name for path in link_dir(home).iterdir()] == ["libnvrtc.so"]


def test_replaces_a_stale_link(clean_env, tmp_path):
    home = fake_toolkit(tmp_path / "cu13", "lib", "libcudart.so.13")
    link = link_dir(home) / "libcudart.so"
    link.unlink()
    link.symlink_to(tmp_path / "elsewhere")
    assert link_dir(home) / "libcudart.so" == link
    assert link.resolve() == (home / "lib" / "libcudart.so.13").resolve()


@pytest.mark.parametrize("lib_dir", ["lib", "lib64"])
def test_leaves_a_linkable_toolkit_alone(clean_env, tmp_path, lib_dir):
    home = fake_toolkit(tmp_path / "cuda", lib_dir, "libcudart.so", "libcudart.so.13")
    assert link_dir(home) is None


def test_leaves_a_toolkit_without_libraries_alone(clean_env, tmp_path):
    assert link_dir(fake_toolkit(tmp_path / "cuda", "lib64")) is None


def test_falls_back_to_the_pip_toolkit(clean_env, monkeypatch, tmp_path):
    home = fake_toolkit(tmp_path / "cu13", "lib", "libcudart.so.13")
    monkeypatch.setattr(cuda_toolkit, "pip_cuda_home", lambda: home)
    monkeypatch.setenv("LIBRARY_PATH", "/opt/lib")
    prepare_cuda_toolkit()
    assert os.environ["CUDA_HOME"] == str(home)
    assert os.environ["LIBRARY_PATH"] == f"{link_dir(home)}{os.pathsep}/opt/lib"


def test_a_configured_toolkit_wins(clean_env, monkeypatch, tmp_path):
    system = fake_toolkit(tmp_path / "system", "lib64", "libcudart.so.13")
    pip = fake_toolkit(tmp_path / "cu13", "lib", "libcudart.so.13")
    monkeypatch.setattr(cuda_toolkit, "pip_cuda_home", lambda: pip)
    monkeypatch.setenv("CUDA_HOME", str(system))
    prepare_cuda_toolkit()
    assert os.environ["CUDA_HOME"] == str(system)
    # The configured toolkit is fixed up too, if it needs it.
    assert os.environ["LIBRARY_PATH"] == str(link_dir(system))


def test_nothing_to_do_without_a_toolkit(clean_env, monkeypatch):
    monkeypatch.setattr(cuda_toolkit, "pip_cuda_home", lambda: None)
    prepare_cuda_toolkit()
    assert "CUDA_HOME" not in os.environ and "LIBRARY_PATH" not in os.environ


def test_a_failure_only_warns(clean_env, monkeypatch, tmp_path):
    home = fake_toolkit(tmp_path / "cu13", "lib", "libcudart.so.13")
    monkeypatch.setattr(cuda_toolkit, "pip_cuda_home", lambda: home)
    blocked = tmp_path / "file"
    blocked.touch()
    monkeypatch.setenv("XDG_CACHE_HOME", str(blocked))
    prepare_cuda_toolkit()
    assert "LIBRARY_PATH" not in os.environ


BUILD = textwrap.dedent(
    """
    import os
    from pathlib import Path

    import torch
    # Imported before the plugin, as vLLM does.
    import torch.utils.cpp_extension as cpp_extension

    assert cpp_extension.CUDA_HOME == os.environ.get("CUDA_HOME")
    from aleph_alpha_inference.kernels.native_build import load_extension

    module = load_extension("cudart_probe", [Path("probe.cu").resolve()])
    x = torch.zeros(4, device="cuda")
    module.run(x)
    assert x.tolist() == [1.0] * 4, x
    """
)

PROBE = textwrap.dedent(
    """
    #include <torch/extension.h>
    __global__ void add_one(float* x) { x[threadIdx.x] += 1.f; }
    void run(torch::Tensor x) { add_one<<<1, x.numel()>>>(x.data_ptr<float>()); }
    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
    """
)


def as_shipped(home: Path, root: Path) -> Path:
    """`home` as its wheels install it, without the `libcudart.so` that may
    have been added to it by hand. Links into `home`, so nothing is copied."""
    root.mkdir()
    for entry in home.iterdir():
        if entry.name != "lib":
            (root / entry.name).symlink_to(entry)
    (root / "lib").mkdir()
    for library in (home / "lib").iterdir():
        if library.name != "libcudart.so":
            (root / "lib" / library.name).symlink_to(library)
    return root


@pytest.mark.gpu
@pytest.mark.parametrize("configured", [True, False], ids=["cuda-home", "unset"])
def test_builds_with_the_pip_toolkit(tmp_path, configured):
    """A fresh process builds with the pip toolkit, as the `cuda` extra
    installs it: with `CUDA_HOME` pointing at it, and without, on a machine
    with no other toolkit."""
    home = pip_cuda_home()
    if home is None:
        pytest.skip("needs the CUDA toolkit from pip (the cuda extra)")
    # The interpreter's directory has `ninja`; the system's for the compiler.
    path = os.pathsep.join([str(Path(sys.executable).parent), "/usr/bin", "/bin"])
    if not configured and (
        shutil.which("nvcc", path=path) or cuda_toolkit.DEFAULT_CUDA_HOME.exists()
    ):
        pytest.skip("a system CUDA toolkit would be used instead")
    (tmp_path / "probe.cu").write_text(PROBE)
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("CUDA_HOME", "CUDA_PATH", "LIBRARY_PATH")
    }
    env |= {
        "PATH": path,
        "XDG_CACHE_HOME": str(tmp_path / "cache"),
        "TORCH_EXTENSIONS_DIR": str(tmp_path / "extensions"),
    }
    if configured:
        env["CUDA_HOME"] = str(as_shipped(home, tmp_path / "cu13"))
    result = subprocess.run(
        [sys.executable, "-c", BUILD],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-4000:]
