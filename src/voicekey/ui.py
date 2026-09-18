"""The window: a status cue, a level meter, and the editable transcript.

Tkinter because it costs nothing -- uv's managed Python bundles Tk 9.0 on every
platform, so there is no `apt install python3-tk`, no system Python, and no
browser engine shipped to draw a rectangle.

Three things here are less obvious than they look.

**The text box is editable on purpose.** Whisper gets technical words wrong
often enough that fixing them in place beats re-dictating, so the UI owns the
transcript and the session only hands it fragments to append. That in turn is
why appending has to preserve the cursor and the scroll position: a burst
finishing while you are mid-correction must not yank the caret to the end.

**The level meter is the only way to notice a dead microphone** before you have
already said the sentence. It is driven from `voicekey.meter`, which is dB
scaled -- speech sits at 0.01-0.1 RMS, so a linear bar looks broken.

**Everything crossing a thread goes through `root.after`.** Tk is not
thread-safe, and the session calls back from its decode worker.
"""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable

from voicekey.display import Display, State

# Status dot colour + caption per state. The dot is the primary cue; the
# caption disambiguates the states that share a colour.
_CUES: dict[State, tuple[str, str]] = {
    "loading": ("#5a5f6a", "Loading model"),
    "idle": ("#5a5f6a", "Ready"),
    "recording": ("#e5484d", "Recording"),
    "transcribing": ("#f5a524", "Transcribing"),
    "done": ("#46a758", "Done"),
    "error": ("#e5484d", "Error"),
}

_BG = "#16181d"
_FG = "#e6e8eb"
_MUTED = "#8b8f98"
_FIELD = "#1e2127"
_METER_BG = "#22262e"
_METER_FG = "#46a758"
_METER_DEAD = "#e5484d"

_HINT = "Ctrl+Shift+D  record / stop        Ctrl+C  copy        Esc  quit"

_METER_W = 180
_METER_H = 10

# Below this the microphone is producing essentially nothing. Silence and a
# muted device look identical on a bar, so the meter says so in words instead.
_DEAD_LEVEL = 0.02


class Popup:
    """A single window. Create once, drive it with `apply` and `append`."""

    def __init__(
        self,
        *,
        on_toggle: Callable[[], None],
        on_quit: Callable[[], None] | None = None,
        backend_label: str = "CPU",
    ) -> None:
        self._on_toggle = on_toggle
        self._on_quit = on_quit
        self._state: State = "loading"

        self.root = tk.Tk()
        self.root.title("voicekey")
        self.root.configure(bg=_BG)
        self.root.geometry("760x460")
        self.root.minsize(520, 300)
        # Wayland will not let a client place itself, but it does honour this.
        self.root.attributes("-topmost", True)

        header = tk.Frame(self.root, bg=_BG)
        header.pack(fill="x", padx=20, pady=(16, 8))

        self._dot = tk.Canvas(header, width=14, height=14, bg=_BG, highlightthickness=0, bd=0)
        self._dot_id = self._dot.create_oval(1, 1, 13, 13, fill=_CUES["loading"][0], outline="")
        self._dot.pack(side="left")

        self._caption = tk.Label(
            header, text=_CUES["loading"][1], bg=_BG, fg=_FG, font=("TkDefaultFont", 13, "bold")
        )
        self._caption.pack(side="left", padx=(10, 14))

        self._meter = tk.Canvas(
            header, width=_METER_W, height=_METER_H, bg=_METER_BG, highlightthickness=0, bd=0
        )
        self._meter_bar = self._meter.create_rectangle(
            0, 0, 0, _METER_H, fill=_METER_FG, outline=""
        )
        self._meter.pack(side="left")

        self._backend = tk.Label(
            header, text=backend_label, bg=_BG, fg=_MUTED, font=("TkDefaultFont", 10)
        )
        self._backend.pack(side="right")

        self._text = tk.Text(
            self.root,
            bg=_FIELD,
            fg=_FG,
            insertbackground=_FG,
            relief="flat",
            wrap="word",
            padx=14,
            pady=12,
            font=("TkDefaultFont", 12),
            highlightthickness=0,
            undo=True,
        )
        self._text.pack(fill="both", expand=True, padx=20)

        footer = tk.Frame(self.root, bg=_BG)
        footer.pack(fill="x", padx=20, pady=(8, 14))

        tk.Label(footer, text=_HINT, bg=_BG, fg=_MUTED, font=("TkDefaultFont", 9)).pack(side="left")

        self._count = tk.Label(footer, text="0 chars", bg=_BG, fg=_MUTED, font=("TkDefaultFont", 9))
        self._count.pack(side="right", padx=(10, 0))

        self._copy_button = tk.Button(
            footer,
            text="Copy",
            command=self.copy,
            bg=_FIELD,
            fg=_FG,
            activebackground=_METER_BG,
            activeforeground=_FG,
            relief="flat",
            highlightthickness=0,
            padx=12,
            takefocus=False,
        )
        self._copy_button.pack(side="right")

        # bind_all, not bind: the text box has focus almost all the time, and a
        # binding on the window alone never fires once a child owns the key.
        self.root.bind_all("<Control-Shift-D>", self._on_hotkey)
        self.root.bind_all("<Control-Shift-d>", self._on_hotkey)
        self.root.bind("<Escape>", self._on_escape)
        self._text.bind("<KeyRelease>", lambda _e: self._refresh_count())
        self.root.protocol("WM_DELETE_WINDOW", self._on_escape)

    # -- input -------------------------------------------------------------

    def _on_hotkey(self, _event: object) -> str:
        self._on_toggle()
        return "break"

    def _on_escape(self, _event: object = None) -> str:
        if self._on_quit is not None:
            self._on_quit()
        self.root.quit()
        return "break"

    # -- the interface a different toolkit would have to satisfy ------------
    #
    # Called from the session's decode worker. Tk is not thread-safe, so every
    # mutation is marshalled onto the main thread via `after`.

    def apply(self, display: Display) -> None:
        """Render a status snapshot."""
        self.root.after(0, self._apply_display, display)

    def append(self, text: str) -> None:
        """Add a finished burst to the transcript."""
        self.root.after(0, self._apply_append, text)

    def set_backend(self, label: str) -> None:
        self.root.after(0, lambda: self._backend.configure(text=label))

    def transcript(self) -> str:
        return self._text.get("1.0", "end").strip()

    def copy(self) -> None:
        """Copy the selection if there is one, otherwise the whole transcript."""
        try:
            text = self._text.get("sel.first", "sel.last")
        except tk.TclError:
            text = self.transcript()
        if not text.strip():
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self._flash("Copied")

    def run(self) -> None:
        self.root.mainloop()

    # -- internals ---------------------------------------------------------

    def _apply_display(self, display: Display) -> None:
        self._state = display.state
        colour, caption = _CUES[display.state]
        self._dot.itemconfigure(self._dot_id, fill=colour)

        if display.message:
            caption = display.message
        elif display.pending:
            # The queue depth is the only sign that work is outstanding, since
            # unfinished text shows nothing at all.
            caption = f"{caption} ({display.pending})"
        self._caption.configure(text=caption)

        self._draw_meter(display.level, recording=display.state == "recording")

    def _draw_meter(self, level: float, *, recording: bool) -> None:
        if not recording:
            self._meter.itemconfigure(self._meter_bar, state="hidden")
            return
        width = max(2, int(_METER_W * min(max(level, 0.0), 1.0)))
        self._meter.coords(self._meter_bar, 0, 0, width, _METER_H)
        self._meter.itemconfigure(
            self._meter_bar,
            state="normal",
            fill=_METER_DEAD if level < _DEAD_LEVEL else _METER_FG,
        )

    def _apply_append(self, text: str) -> None:
        """Append without disturbing an edit in progress.

        The caret and the scroll position belong to whatever the user is doing
        right now; a burst landing mid-correction must not move either.
        """
        was_at_end = self._text.compare("insert", "==", "end-1c")
        top, _bottom = self._text.yview()
        caret = self._text.index("insert")

        existing = self._text.get("1.0", "end").strip()
        self._text.insert("end", (" " if existing else "") + text)

        if was_at_end:
            self._text.see("end")
        else:
            self._text.mark_set("insert", caret)
            self._text.yview_moveto(top)
        self._refresh_count()

    def _refresh_count(self) -> None:
        self._count.configure(text=f"{len(self.transcript())} chars")

    def _flash(self, message: str) -> None:
        """Briefly confirm an action, then fall back to the state's caption --
        copying is otherwise completely invisible."""
        self._caption.configure(text=message)
        self.root.after(900, lambda: self._caption.configure(text=_CUES[self._state][1]))
