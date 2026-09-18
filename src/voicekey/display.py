"""The one type the UI and the session agree on.

This is the seam of the whole app. `Session` produces `Display`; the UI
consumes it and knows nothing else -- not audio, not numpy, not the model.
Keeping it a frozen dataclass rather than a pile of setters means the UI can
never observe a half-updated state, which matters because the session runs on
a worker thread and Tk redraws on the main one.

Deliberately absent: the transcript itself. The text box is editable, so the
*UI* owns the text and the session only ever hands it new material to append
(see `Session.on_append`). If the session also owned the transcript, every
burst that finished mid-edit would have to reconcile against whatever the user
had just typed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

State = Literal["loading", "idle", "recording", "transcribing", "done", "error"]


@dataclass(frozen=True)
class Display:
    """Everything the status line shows, as one immutable snapshot."""

    state: State = "loading"

    # Mic level, 0..1, already log-scaled for display (see voicekey.meter).
    # Only meaningful while recording; otherwise 0.0.
    level: float = 0.0

    # Bursts recorded but not yet transcribed, including the one in flight.
    # This is the only cue that work is outstanding, since finished text just
    # appears and unfinished text shows nothing at all.
    pending: int = 0

    # Short human-readable line: an error, or a note like "Loading model".
    # Empty means "nothing worth saying", and the state's own caption shows.
    message: str = ""

    def with_(self, **changes: object) -> Display:
        """A new snapshot with some fields changed."""
        return replace(self, **changes)


__all__ = ["Display", "State"]
