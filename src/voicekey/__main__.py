"""Entry point.

Scaffold stage: this wires the UI state machine and reports the resolved
backend, so the packaging and CI story can be proven end to end before the
engine exists. Recording and transcription land next (see STATE.md).
"""

from __future__ import annotations

import sys

from voicekey import backend as backend_mod
from voicekey.paths import models_dir
from voicekey.ui import Popup


def main() -> int:
    be = backend_mod.resolve()

    state = {"recording": False}
    popup: Popup

    def on_toggle() -> None:
        if state["recording"]:
            state["recording"] = False
            popup.set_state("transcribing")
            # TODO(engine): decode the captured buffer.
            popup.set_state("done")
            popup.set_text("(engine not wired up yet)")
        else:
            state["recording"] = True
            popup.set_text("")
            popup.set_state("recording")

    popup = Popup(on_toggle=on_toggle, backend_label=be.label)
    popup.set_text(
        f"backend   {be.label}  ({be.quantization})\n"
        f"providers {', '.join(be.providers)}\n"
        f"models    {models_dir()}"
    )
    popup.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
