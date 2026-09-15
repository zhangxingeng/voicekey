"""Microphone capture at the rate Whisper wants.

PortAudio is asked for 16kHz mono f32 directly rather than resampling
ourselves -- it will negotiate with the device and convert, which is one less
piece of DSP to get subtly wrong.

Capture appends into a list of blocks from PortAudio's own callback thread, so
nothing in the hot path allocates a growing array or holds a lock for long.
`stop()` is what concatenates.
"""

from __future__ import annotations

import threading

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16_000

# PulseAudio/PipeWire expose `Monitor of ...` loopback sources that record
# system *output*. They enumerate alongside real microphones, so a naive
# "first input device" pick can silently record the speakers instead of the
# user. Filter them out of anything shown as a microphone choice.
_LOOPBACK_MARKERS = ("monitor of", "loopback")


def input_devices() -> list[tuple[int, str]]:
    """Real microphones: (index, name), loopback monitors excluded."""
    out = []
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] < 1:
            continue
        if any(marker in dev["name"].lower() for marker in _LOOPBACK_MARKERS):
            continue
        out.append((i, dev["name"]))
    return out


class Recorder:
    """One capture session. Not reusable -- construct per recording."""

    def __init__(self, device: int | None = None) -> None:
        self._blocks: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=device,
            callback=self._on_audio,
        )

    def _on_audio(self, indata: np.ndarray, _frames: int, _time: object, status: object) -> None:
        if status:
            # Overflows are recoverable and not worth interrupting a recording
            # for; the dropped frames are already gone either way.
            print(f"[audio] {status}", flush=True)
        with self._lock:
            self._blocks.append(indata[:, 0].copy())

    def start(self) -> None:
        self._stream.start()

    def stop(self) -> np.ndarray:
        """Stop capture and return everything recorded, as 16kHz mono f32."""
        self._stream.stop()
        self._stream.close()
        with self._lock:
            if not self._blocks:
                return np.zeros(0, dtype=np.float32)
            return np.concatenate(self._blocks)

    @property
    def seconds(self) -> float:
        with self._lock:
            return sum(len(b) for b in self._blocks) / SAMPLE_RATE
