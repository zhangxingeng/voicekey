"""The popup: a status cue and a text box. Nothing else, deliberately.

Tkinter because it costs nothing -- uv's Python bundles Tk 9.0 on every
platform, so there is no `apt install python3-tk`, no system Python, and no
browser engine shipped to draw a rectangle. Startup is ~40ms, which matters
because the window is mapped on every hotkey press.

The whole UI is kept behind `Popup`'s four methods so swapping to GTK4/Qt
later is an afternoon, not a rewrite.
"""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from typing import Literal

State = Literal["idle", "recording", "transcribing", "done", "error"]

# Status dot colour + caption per state. The dot is the primary cue; the
# caption is there for the states where colour alone is ambiguous.
_CUES: dict[State, tuple[str, str]] = {
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


class Popup:
    """A single small window. Create once, show/hide per session."""

    def __init__(self, *, on_toggle: Callable[[], None], backend_label: str = "CPU") -> None:
        self._on_toggle = on_toggle
        self._state: State = "idle"

        self.root = tk.Tk()
        self.root.title("voicekey")
        self.root.configure(bg=_BG)
        self.root.geometry("460x240")
        self.root.minsize(360, 180)

        header = tk.Frame(self.root, bg=_BG)
        header.pack(fill="x", padx=14, pady=(12, 8))

        self._dot = tk.Canvas(header, width=12, height=12, bg=_BG, highlightthickness=0, bd=0)
        self._dot_id = self._dot.create_oval(1, 1, 11, 11, fill=_CUES["idle"][0], outline="")
        self._dot.pack(side="left")

        self._caption = tk.Label(
            header, text=_CUES["idle"][1], bg=_BG, fg=_FG, font=("TkDefaultFont", 11, "bold")
        )
        self._caption.pack(side="left", padx=(8, 0))

        tk.Label(header, text=backend_label, bg=_BG, fg=_MUTED, font=("TkDefaultFont", 9)).pack(
            side="right"
        )

        self._text = tk.Text(
            self.root,
            bg=_FIELD,
            fg=_FG,
            insertbackground=_FG,
            relief="flat",
            wrap="word",
            height=6,
            padx=10,
            pady=8,
            font=("TkDefaultFont", 10),
        )
        self._text.pack(fill="both", expand=True, padx=14)

        footer = tk.Frame(self.root, bg=_BG)
        footer.pack(fill="x", padx=14, pady=10)

        self._toggle_btn = tk.Button(
            footer,
            text="Record  (Space)",
            command=self._on_toggle,
            relief="flat",
            bg="#2a2f38",
            fg=_FG,
            activebackground="#343a45",
            activeforeground=_FG,
            padx=14,
            pady=5,
            bd=0,
            highlightthickness=0,
        )
        self._toggle_btn.pack(side="left")

        tk.Button(
            footer,
            text="Copy",
            command=self.copy,
            relief="flat",
            bg="#2a2f38",
            fg=_FG,
            activebackground="#343a45",
            activeforeground=_FG,
            padx=14,
            pady=5,
            bd=0,
            highlightthickness=0,
        ).pack(side="right")

        self.root.bind("<space>", lambda _e: self._on_toggle())
        self.root.bind("<Escape>", lambda _e: self.root.quit())

    # -- the four-method interface a different toolkit would have to satisfy --

    def set_state(self, state: State) -> None:
        self._state = state
        colour, caption = _CUES[state]
        self._dot.itemconfigure(self._dot_id, fill=colour)
        self._caption.configure(text=caption)
        self._toggle_btn.configure(
            text="Stop  (Space)" if state == "recording" else "Record  (Space)"
        )
        self.root.update_idletasks()

    def set_text(self, text: str) -> None:
        self._text.delete("1.0", "end")
        self._text.insert("1.0", text)

    def copy(self) -> None:
        text = self._text.get("1.0", "end").strip()
        if not text:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)

    def run(self) -> None:
        self.root.mainloop()
