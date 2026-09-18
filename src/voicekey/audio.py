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


# PulseAudio/PipeWire expose loopback sources that record system *output*.
# They enumerate alongside real microphones, so a naive "first input device"
# pick can silently record the speakers instead of the user.
#
# The same device appears under two different spellings depending on who is
# asking: PulseAudio's *description* is "Monitor of <sink>", while PortAudio
# reports the PipeWire *node name*, where it is a ".monitor" suffix
# (alsa_output.usb-....analog-stereo.monitor). Both forms must be matched --
# checking only the description silently lets every monitor through.
def _is_loopback(name: str) -> bool:
    lowered = name.lower()
    return (
        lowered.startswith("monitor of")
        or lowered.endswith(".monitor")
        or ".monitor." in lowered
        or "loopback" in lowered
    )


def input_devices() -> list[tuple[int, str]]:
    """Real microphones: (index, name), loopback monitors excluded."""
    return [
        (i, dev["name"])
        for i, dev in enumerate(sd.query_devices())
        if dev["max_input_channels"] >= 1 and not _is_loopback(dev["name"])
    ]


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

    def latest(self, frames: int = SAMPLE_RATE // 10) -> np.ndarray:
        """The most recently captured audio, for the level meter.

        Reads from the tail rather than concatenating everything: the meter is
        polled several times a second, and by the end of a long recording
        joining the whole buffer each time would cost more than the decode.
        """
        with self._lock:
            tail: list[np.ndarray] = []
            total = 0
            for block in reversed(self._blocks):
                tail.append(block)
                total += len(block)
                if total >= frames:
                    break
        if not tail:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(list(reversed(tail)))[-frames:]
