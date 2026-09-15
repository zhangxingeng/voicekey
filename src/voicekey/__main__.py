"""Entry point: record, transcribe, show the text.

The model is loaded once on a background thread at startup and kept warm --
loading takes seconds, and the whole premise is that recording begins the
instant the window appears. Transcription also runs off the UI thread, so the
status cue keeps updating while the decode runs.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

from voicekey import backend as backend_mod
from voicekey.audio import Recorder
from voicekey.engine import Whisper
from voicekey.paths import models_dir
from voicekey.ui import Popup

MODEL_SUBDIR = "sherpa-onnx-whisper-turbo"


def _find_model_dir() -> Path | None:
    candidates = [
        models_dir() / MODEL_SUBDIR,
        # The previous Tauri app downloaded the same artifacts; reuse them
        # rather than making the user fetch 1GB again.
        Path.home() / ".prompt-compose" / "models" / MODEL_SUBDIR,
    ]
    return next((c for c in candidates if (c / "turbo-tokens.txt").is_file()), None)


def main() -> int:
    model_dir = _find_model_dir()
    if model_dir is None:
        print(
            f"No model found. Expected it in {models_dir() / MODEL_SUBDIR}.\n"
            "Fetch it with: ./salvage/download-whisper-model.sh",
            file=sys.stderr,
        )
        return 1

    state: dict[str, object] = {"recorder": None, "model": None}
    popup: Popup

    def load_model() -> None:
        be = backend_mod.resolve()
        model = Whisper(model_dir, be)
        state["model"] = model
        popup.set_backend(model.backend.label)
        popup.set_state("idle")
        popup.set_text("Press Space to start recording.")

    def transcribe(samples) -> None:
        model: Whisper = state["model"]  # type: ignore[assignment]
        try:
            result = model.transcribe(samples)
            # Empty means the silence gate rejected it (voicekey.vad) -- say so
            # rather than showing a blank box that looks like a failure.
            popup.set_text(result.text or "No speech detected.")
            popup.set_state("done" if result.text else "idle")
        except Exception as exc:  # surfaced in the box, never swallowed
            popup.set_text(f"Transcription failed: {exc}")
            popup.set_state("error")

    def on_toggle() -> None:
        if state["model"] is None:
            return  # still loading; the cue already says so

        recorder = state["recorder"]
        if recorder is None:
            rec = Recorder()
            rec.start()
            state["recorder"] = rec
            popup.set_text("")
            popup.set_state("recording")
        else:
            samples = recorder.stop()  # type: ignore[union-attr]
            state["recorder"] = None
            popup.set_state("transcribing")
            popup.set_text(f"{len(samples) / 16000:.1f}s captured, decoding...")
            threading.Thread(target=transcribe, args=(samples,), daemon=True).start()

    popup = Popup(on_toggle=on_toggle, backend_label="loading")
    popup.set_state("loading")
    popup.set_text("Loading model...")
    threading.Thread(target=load_model, daemon=True).start()
    popup.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
