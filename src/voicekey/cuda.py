"""Make ONNX Runtime's CUDA provider loadable from pip-installed NVIDIA wheels.

The problem this solves: `onnxruntime-gpu` does *not* bundle the CUDA runtime.
Its `libonnxruntime_providers_cuda.so` links against libcublasLt, libcudnn,
libcurand and friends, and if the dynamic loader cannot find them the provider
fails to load. ONNX Runtime then **silently falls back to CPU** -- inference
still produces correct output, just ~2x slower, with nothing but a warning on
stderr to say so.

The NVIDIA pip wheels do ship those libraries, but CUDA 13 changed the layout
to a shared `nvidia/cu13/lib` directory (CUDA 12 used a per-package
`nvidia/<name>/lib`), and ONNX Runtime 1.30 does not search the new location.
So we dlopen them ourselves, with RTLD_GLOBAL, before any session is created --
which puts the symbols in the global namespace where the provider's own load
will find them.

This is a no-op on machines with no NVIDIA wheels installed, so it is always
safe to call.
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

# Load order matters: libraries that others link against come first.
_PRELOAD = (
    "libcudart.so",
    "libnvJitLink.so",
    "libcublasLt.so",
    "libcublas.so",
    "libcufft.so",
    "libcurand.so",
    "libcusparse.so",
    "libcusolver.so",
    "libcudnn.so",
)

_done = False


def _nvidia_lib_dirs() -> list[Path]:
    dirs: list[Path] = []
    for entry in sys.path:
        nvidia = Path(entry) / "nvidia"
        if not nvidia.is_dir():
            continue
        # CUDA 13 consolidates into nvidia/cu13/lib; CUDA 12 used
        # nvidia/<component>/lib. Glob covers both without special-casing.
        dirs.extend(p for p in nvidia.glob("*/lib") if p.is_dir())
    return dirs


def preload() -> list[str]:
    """dlopen the NVIDIA runtime libraries. Returns the ones loaded.

    Idempotent and never raises -- a missing library just means the CUDA
    provider will not be available, which the caller detects by inspecting
    `session.get_providers()` afterwards.
    """
    global _done
    if _done:
        return []
    _done = True

    if not sys.platform.startswith("linux"):
        return []

    loaded: list[str] = []
    dirs = _nvidia_lib_dirs()
    for stem in _PRELOAD:
        for directory in dirs:
            matches = sorted(directory.glob(f"{stem}*"))
            if not matches:
                continue
            try:
                ctypes.CDLL(str(matches[-1]), mode=ctypes.RTLD_GLOBAL)
                loaded.append(matches[-1].name)
            except OSError:
                pass
            break
    return loaded
