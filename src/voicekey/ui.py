"""The popup: a status cue and the text. Keyboard only, no buttons.

Space records, Ctrl+C copies, Escape quits. Buttons were tried first and
removed -- they duplicated shortcuts the user reaches for anyway, and every
control on screen is one more thing to read before saying a word.

Tkinter because it costs nothing -- uv's Python bundles Tk 9.0 on every
platform, so there is no `apt install python3-tk`, no system Python, and no
browser engine shipped to draw a rectangle. Startup is ~40ms, which matters
because the window is mapped on every hotkey press.

The whole UI sits behind `Popup`'s four setters so swapping to GTK4/Qt later is
an afternoon, not a rewrite.
"""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from typing import Literal

State = Literal["loading", "idle", "recording", "transcribing", "done", "error"]

# Status dot colour + caption per state. The dot is the primary cue; the
# caption is there for the states where colour alone is ambiguous.
_CUES: dict[State, tuple[str, str]] = {
    "loading": ("#5a5f6a", "Loading model"),
    "idle": ("#5a5f6a", "Ready"),
    "recording": ("#e5484d", "Recording"),
    "transcribing": ("#f5a524", "Transcribing..."),
    "done": ("#46a758", "Done"),
    "error": ("#e5484d", "Error"),
}

_BG = "#16181d"
_FG = "#e6e8eb"
_MUTED = "#8b8f98"
_FIELD = "#1e2127"

_HINT = "Space  record / stop        Ctrl+C  copy        Esc  quit"


class Popup:
    """A single window. Create once, drive it with the setters."""

    def __init__(self, *, on_toggle: Callable[[], None], backend_label: str = "CPU") -> None:
        self._on_toggle = on_toggle
        self._state: State = "idle"

        self.root = tk.Tk()
        self.root.title("voicekey")
        self.root.configure(bg=_BG)
        self.root.geometry("760x460")
        self.root.minsize(520, 300)

        header = tk.Frame(self.root, bg=_BG)
        header.pack(fill="x", padx=20, pady=(16, 10))

        self._dot = tk.Canvas(header, width=14, height=14, bg=_BG, highlightthickness=0, bd=0)
        self._dot_id = self._dot.create_oval(1, 1, 13, 13, fill=_CUES["idle"][0], outline="")
        self._dot.pack(side="left")

        self._caption = tk.Label(
            header, text=_CUES["idle"][1], bg=_BG, fg=_FG, font=("TkDefaultFont", 13, "bold")
        )
        self._caption.pack(side="left", padx=(10, 0))

        self._backend = tk.Label(
            header, text=backend_label, bg=_BG, fg=_MUTED, font=("TkDefaultFont", 10)
        )
        self._backend.pack(side="right")

        self._text = tk.Text(
            self.root,
            bg=_FIELD,
            fg=_FG,
            relief="flat",
            wrap="word",
            padx=14,
            pady=12,
            font=("TkDefaultFont", 12),
            # Read-only, so Space reaches the window binding instead of typing
            # a space. Selection and Ctrl+C still work on a disabled Text.
            state="disabled",
            highlightthickness=0,
        )
        self._text.pack(fill="both", expand=True, padx=20)

        tk.Label(self.root, text=_HINT, bg=_BG, fg=_MUTED, font=("TkDefaultFont", 9)).pack(
            fill="x", padx=20, pady=(10, 14)
        )

        self.root.bind("<space>", self._on_space)
        self.root.bind("<Control-c>", self._on_copy)
        self.root.bind("<Escape>", lambda _e: self.root.quit())

    # -- input -------------------------------------------------------------

    def _on_space(self, _event: object) -> str:
        self._on_toggle()
        return "break"

    def _on_copy(self, _event: object) -> str:
        self.copy()
        return "break"

    # -- the interface a different toolkit would have to satisfy -------------
    #
    # These are called from worker threads (model loading, decoding). Tk is not
    # thread-safe, so every mutation is marshalled onto the main thread via
    # `after`, which is the one cross-thread call Tk does support.

    def set_state(self, state: State) -> None:
        self.root.after(0, self._apply_state, state)

    def set_backend(self, label: str) -> None:
        self.root.after(0, lambda: self._backend.configure(text=label))

    def set_text(self, text: str) -> None:
        self.root.after(0, self._apply_text, text)

    def copy(self) -> None:
        """Copy the selection if there is one, otherwise the whole transcript."""
        try:
            text = self._text.get("sel.first", "sel.last")
        except tk.TclError:
            text = self._text.get("1.0", "end")
        text = text.strip()
        if not text:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self._flash("Copied")

    def run(self) -> None:
        self.root.mainloop()

    # -- internals ---------------------------------------------------------

    def _apply_state(self, state: State) -> None:
        self._state = state
        colour, caption = _CUES[state]
        self._dot.itemconfigure(self._dot_id, fill=colour)
        self._caption.configure(text=caption)

    def _apply_text(self, text: str) -> None:
        self._text.configure(state="normal")
        self._text.delete("1.0", "end")
        self._text.insert("1.0", text)
        self._text.configure(state="disabled")

    def _flash(self, message: str) -> None:
        """Briefly confirm an action, then fall back to the current state's
        caption -- copying is otherwise completely invisible."""
        self._caption.configure(text=message)
        self.root.after(900, lambda: self._caption.configure(text=_CUES[self._state][1]))
