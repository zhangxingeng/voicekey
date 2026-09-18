"""Which execution provider and model quantization to use.

Design rule: the provider is a *runtime probe*, never an install-time
assumption. The same artifact runs on a CUDA box, an Apple Silicon laptop, and
a machine with no accelerator at all -- it just asks ONNX Runtime what is
actually available and picks the best option present, always with
CPUExecutionProvider appended as the real fallback.

On quantization: the plan was to load fp16 weights whenever an accelerator was
present, on the theory that ORT's CUDA EP handles quantized ops
(MatMulInteger/QuantizeLinear) badly enough to make int8-on-CUDA slower than
int8-on-CPU. Measured on an RTX 3090 Ti, that turned out to be wrong -- int8 on
CUDA runs 2.15x faster than the same model on CPU (2.71x vs 1.26x realtime on
an 11s clip).

So there is one model for every backend: int8. That halves the first-run
download and removes a whole axis of configuration. Whether fp16 would be
faster still is untested and deliberately left as a later optimization, which
is why this field exists rather than being inlined.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Present exactly when the NVIDIA kernel driver is loaded. Used only to tell
# "this machine has no GPU" apart from "this machine has a GPU we failed to
# use" -- the second deserves to be said out loud, because the app keeps
# working and is merely 2x slower, which is easy to live with unknowingly for
# weeks. Installing without the `cuda` extra lands you here silently.
_NVIDIA_DRIVER = Path("/proc/driver/nvidia/version")

# Ordered best-first. Each entry is (provider name, human label).
_ACCELERATED: tuple[tuple[str, str], ...] = (
    ("CUDAExecutionProvider", "CUDA"),
    ("CoreMLExecutionProvider", "CoreML"),
    ("DmlExecutionProvider", "DirectML"),
)

_CPU = "CPUExecutionProvider"


@dataclass(frozen=True)
class Backend:
    """The resolved inference configuration."""

    providers: list[str]
    """Passed straight to ort.InferenceSession(providers=...)."""

    quantization: str
    """'fp16' when accelerated, 'int8' on CPU -- selects which model files to load."""

    label: str
    """Short human-readable name for the UI, e.g. 'CUDA' or 'CPU'."""

    @property
    def accelerated(self) -> bool:
        return self.providers[0] != _CPU


def resolve(available: list[str] | None = None) -> Backend:
    """Pick the best backend from what ONNX Runtime reports as available.

    `available` is injectable so this stays testable without a GPU (and without
    importing onnxruntime at all).
    """
    if available is None:
        import onnxruntime as ort

        available = ort.get_available_providers()

    for provider, label in _ACCELERATED:
        if provider in available:
            return Backend(providers=[provider, _CPU], quantization="int8", label=label)

    return Backend(providers=[_CPU], quantization="int8", label="CPU")


def nvidia_present(driver: Path = _NVIDIA_DRIVER) -> bool:
    """True when this machine has a loaded NVIDIA driver.

    Linux only, and deliberately so: this exists to catch a wasted GPU on the
    development machine, not to gate anything. Elsewhere it returns False and
    nothing is claimed.
    """
    return driver.exists()


def missing_acceleration(backend: Backend, *, driver: Path = _NVIDIA_DRIVER) -> bool:
    """True when there is a GPU here that we are not using."""
    return not backend.accelerated and nvidia_present(driver)


def confirm(session_providers: list[str]) -> Backend:
    """Re-resolve from what a *live session* actually bound.

    `get_available_providers()` reports what ONNX Runtime was *built* with, not
    what it can load. onnxruntime-gpu happily lists CUDAExecutionProvider on a
    machine with no CUDA runtime installed, then silently falls back to CPU at
    session-creation time -- the transcription still works, it is just 20x
    slower than the UI claims.

    So the label shown to the user comes from `session.get_providers()` after
    the fact, never from the pre-flight probe.
    """
    return resolve(session_providers)
