"""Entry point: wire the hotkey, the recorder, the model and the window together.

Two modes, one binary. Run bare, it is the app. Run with `--toggle`, it is the
tiny client GNOME executes when the hotkey fires, which does nothing but poke
the running app over its socket and exit.

The ordering here is the product. Audio capture opens in ~20ms, a Tk window
maps in ~50-100ms, and the model takes seconds -- so the window appears and
recording starts without waiting for any of it, and bursts queue up until the
model is ready to drain them.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

from voicekey import backend as backend_mod
from voicekey import hotkey, ipc, meter, models
from voicekey.audio import Recorder
from voicekey.display import Display
from voicekey.engine import Whisper
from voicekey.paths import models_dir
from voicekey.session import Session
from voicekey.ui import Popup

# How often the meter is redrawn while recording. Fast enough to look live,
# slow enough that it never competes with the decode for the GIL.
_METER_MS = 60


def _toggle() -> int:
    """The `--toggle` mode: tell the running app to start or stop recording."""
    reply = ipc.send(ipc.TOGGLE)
    if reply is None:
        print("voicekey is not running", file=sys.stderr)
        return 1
    return 0


def _launcher_command() -> str:
    """The command GNOME should run when the hotkey fires.

    Frozen, sys.executable *is* the app. Running from a checkout it is the
    interpreter, which needs the module spelled out.
    """
    if getattr(sys, "frozen", False):
        return f"{sys.executable} --toggle"
    return f"{sys.executable} -m voicekey --toggle"


def _find_model(on_progress) -> Path:
    """The model directory, downloading it on first run.

    The previous Tauri app left the same artifacts behind; reuse them rather
    than making the user fetch a gigabyte they already have.
    """
    legacy = Path.home() / ".prompt-compose" / "models" / models.MODEL_SUBDIR
    if (legacy / "turbo-tokens.txt").is_file():
        return legacy
    return models.ensure_model(models_dir(), on_progress=on_progress)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--toggle" in argv:
        return _toggle()

    state: dict[str, object] = {"recorder": None}

    popup = Popup(on_toggle=lambda: on_toggle(), on_quit=lambda: shutdown(), backend_label="…")
    session = Session(on_append=popup.append, on_change=popup.apply)

    # -- the model, loaded off the UI thread ------------------------------

    def report(done: int, total: int) -> None:
        share = f"{done / total:.0%}" if total else f"{done / 1e6:.0f} MB"
        popup.apply(Display(state="loading", message=f"Downloading model {share}"))

    def load_model() -> None:
        try:
            model_dir = _find_model(report)
            model = Whisper(model_dir, backend_mod.resolve())
            popup.set_backend(model.backend.label)
            session.set_transcribe(lambda samples: model.transcribe(samples).text)
            if backend_mod.missing_acceleration(model.backend):
                # Silently running on CPU next to an idle GPU is only about 2x
                # slower, which is easy to put up with for weeks without
                # realising. Say it once, in the place the user is looking.
                popup.apply(
                    Display(state="idle", message="GPU found but unused — install the cuda extra")
                )
        except Exception as exc:
            popup.apply(Display(state="error", message=f"Model failed to load: {exc}"))

    # -- recording --------------------------------------------------------

    def pump_meter() -> None:
        recorder = state.get("recorder")
        if recorder is not None:
            session.set_level(meter.level(recorder.latest()))  # type: ignore[union-attr]
        popup.root.after(_METER_MS, pump_meter)

    def on_toggle() -> None:
        recorder = state.get("recorder")
        if recorder is None:
            rec = Recorder()
            rec.start()
            state["recorder"] = rec
            session.set_recording(True)
        else:
            state["recorder"] = None
            session.set_recording(False)
            session.set_level(0.0)
            session.submit(recorder.stop())  # type: ignore[union-attr]

    # -- the hotkey -------------------------------------------------------

    server = ipc.Server(lambda command: _handle(command))

    def _handle(command: str) -> str:
        if command == ipc.PING:
            return ipc.OK
        if command == ipc.TOGGLE:
            # Marshal onto the Tk thread: this runs on the socket's thread.
            popup.root.after(0, on_toggle)
            return ipc.OK
        return ipc.ERROR

    def shutdown() -> None:
        session.close()
        server.close()

    try:
        server.start()
    except RuntimeError:
        print("voicekey is already running", file=sys.stderr)
        return 1

    if hotkey.available():
        try:
            hotkey.register(_launcher_command())
        except Exception as exc:
            popup.apply(Display(state="idle", message=f"Hotkey not registered: {exc}"))
    else:
        # Not an error: the app works, it just has no global key here.
        popup.apply(Display(state="loading", message="No global hotkey on this desktop"))

    threading.Thread(target=load_model, daemon=True).start()
    popup.root.after(_METER_MS, pump_meter)
    try:
        popup.run()
    finally:
        shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
