"""Which execution provider and model quantization to use.

Design rule: the provider is a *runtime probe*, never an install-time
assumption. The same artifact runs on a CUDA box, an Apple Silicon laptop, and
a machine with no accelerator at all -- it just asks ONNX Runtime what is
actually available and picks the best option present, always with
CPUExecutionProvider appended as the real fallback.

Quantization follows from the probe rather than being configured separately:
int8 is the right choice on CPU, but ORT's CUDA EP handles quantized ops
(MatMulInteger/QuantizeLinear) poorly and can silently fall back to CPU
per-node -- which makes int8-on-CUDA *slower* than int8-on-CPU. So an
accelerated provider implies the fp16 weights.
"""

from __future__ import annotations

from dataclasses import dataclass

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
            return Backend(providers=[provider, _CPU], quantization="fp16", label=label)

    return Backend(providers=[_CPU], quantization="int8", label="CPU")
