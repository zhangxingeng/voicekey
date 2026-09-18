"""Recording bursts in, transcript fragments out, in order.

This is the coordination layer, and the reason it exists is that recording and
transcription run at very different speeds: a 30s burst decodes in ~11s on GPU,
so making the user wait before speaking again would be the app's worst flaw.
Bursts are queued and decoded serially in the background while the user keeps
talking.

Two things this has to get right:

  * **Never block recording.** Bursts may be submitted before the model has
    finished loading, and before any previous burst has finished decoding.
  * **Shut down promptly even when the model never arrived.** The user can quit
    while the model is still loading with bursts already queued, and the app
    has to exit anyway.

Ordering is guaranteed by construction rather than by bookkeeping: one worker
thread decodes one burst at a time, so completions cannot interleave. An
earlier version slotted results by sequence number to reorder them, which was
untestable dead weight -- with a serial worker there is nothing to reorder, and
the test that claimed to prove otherwise could not produce two overlapping
decodes at all.

Transcription is injected rather than imported so this file can be tested in
milliseconds with a fake, without a 1GB model or a GPU.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable

import numpy as np

from voicekey.display import Display, State

# Given the samples of one burst, return its text. Raising is allowed: the
# session reports it via Display.message and carries on with the next burst.
TranscribeFn = Callable[[np.ndarray], str]

# Called on the worker thread, once per burst, in submission order, with the
# text to append. The UI owns the transcript because it is user-editable, so
# the session never holds it.
AppendFn = Callable[[str], None]

# Called whenever any field of the snapshot changes.
ChangeFn = Callable[[Display], None]

# How long the worker sleeps between checks while waiting for the model or for
# a burst. Short enough that quitting feels instant, cheap enough to ignore.
_POLL_S = 0.05


class Session:
    """Owns the burst queue and the status snapshot. Thread-safe.

    Callbacks are invoked outside the state lock, so `on_append` and `on_change`
    must not call back into this object synchronously. The real UI satisfies
    this by marshalling onto the Tk main loop via `root.after`.
    """

    def __init__(self, *, on_append: AppendFn, on_change: ChangeFn) -> None:
        self._on_append = on_append
        self._on_change = on_change

        self._lock = threading.Lock()
        # Held across "capture snapshot, hand it to on_change" so two threads
        # cannot deliver their snapshots in the opposite order to the mutations
        # that produced them, which would make the UI flicker backwards.
        # Separate from _lock precisely so a callback never runs under it.
        self._notify_lock = threading.Lock()

        self._snapshot = Display(state="loading")
        self._recording = False
        self._transcribe_fn: TranscribeFn | None = None

        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._sequence = 0

        self._model_ready = threading.Event()
        self._shutdown = threading.Event()

        # daemon=True is a backstop, not the mechanism: close() stops the
        # worker properly. It means a bug in that path can never wedge process
        # exit, which in a GUI app looks like a hang the user has to force-kill.
        self._worker = threading.Thread(target=self._run, daemon=True, name="voicekey-decode")
        self._worker.start()

    # -- wiring ------------------------------------------------------------

    def set_transcribe(self, fn: TranscribeFn) -> None:
        """Supply the model once it has finished loading in the background.

        Any bursts recorded before this point start draining immediately.
        """
        with self._lock:
            self._transcribe_fn = fn
        self._model_ready.set()
        self._publish()

    # -- recording ---------------------------------------------------------

    def set_recording(self, recording: bool) -> None:
        """Update the state cue. Does not itself start or stop audio capture."""
        with self._lock:
            self._recording = recording
        self._publish()

    def set_level(self, level: float) -> None:
        """Push a new mic level (0..1) for the bar. Called frequently."""
        with self._lock:
            if self._snapshot.level == level:
                return
            self._snapshot = self._snapshot.with_(level=level)
        self._publish()

    def submit(self, samples: np.ndarray) -> int:
        """Queue one finished burst for transcription. Returns its sequence number."""
        with self._lock:
            sequence = self._sequence
            self._sequence += 1
        self._queue.put(samples)
        self._publish(pending_delta=1)
        return sequence

    # -- lifecycle ---------------------------------------------------------

    @property
    def display(self) -> Display:
        """The current snapshot."""
        with self._lock:
            return self._snapshot

    def close(self) -> None:
        """Stop the worker and wait for it. Idempotent.

        Queued bursts are abandoned rather than drained. Draining would block
        quit on a model that may never load, to produce text that has nowhere
        left to go because the window is closing.
        """
        self._shutdown.set()
        # Unblocks the worker if it is parked in queue.get().
        self._queue.put(np.empty(0, dtype=np.float32))
        if self._worker.is_alive():
            self._worker.join(timeout=2.0)

    # -- internals ---------------------------------------------------------

    def _compute_state(self, pending: int, *, error: bool) -> State:
        if error:
            return "error"
        if self._recording:
            return "recording"
        if self._transcribe_fn is None:
            return "loading"
        if pending > 0:
            return "transcribing"
        return "idle"

    def _publish(
        self,
        *,
        pending_delta: int = 0,
        error: str | None = None,
        appended: bool = False,
    ) -> None:
        """Recompute the snapshot, then hand it to on_change outside the lock."""
        with self._lock:
            pending = self._snapshot.pending + pending_delta
            state = self._compute_state(pending, error=error is not None)
            if state == "idle" and appended:
                state = "done"
            self._snapshot = self._snapshot.with_(
                pending=pending,
                state=state,
                # A stale error must not outlive the state that produced it,
                # or the box keeps reporting a failure already superseded.
                message=error or "",
            )
            snapshot = self._snapshot

        with self._notify_lock:
            self._on_change(snapshot)

    def _run(self) -> None:
        while not self._shutdown.is_set():
            try:
                samples = self._queue.get(timeout=_POLL_S)
            except queue.Empty:
                continue
            if self._shutdown.is_set():
                return

            transcribe = self._await_model()
            if transcribe is None:
                return

            text, error = "", None
            try:
                text = transcribe(samples)
            except Exception as exc:
                error = str(exc)

            # Append before publishing, so the text is on screen by the time
            # the status claims the queue has drained.
            if text:
                self._on_append(text)
            self._publish(pending_delta=-1, error=error, appended=bool(text))

    def _await_model(self) -> TranscribeFn | None:
        """Block until the model is supplied. None means shutdown won."""
        while not self._model_ready.wait(_POLL_S):
            if self._shutdown.is_set():
                return None
        with self._lock:
            return self._transcribe_fn


__all__ = ["AppendFn", "ChangeFn", "Session", "TranscribeFn"]
