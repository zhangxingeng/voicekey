"""Tests for the recording/transcription coordination layer.

Everything here uses a fake transcribe function -- the point of injecting it is
that this file needs no model, no GPU and no audio device, and runs in well
under a second.

Waiting is always done by polling a predicate with a timeout, never by a bare
sleep. A fixed sleep that happens to be long enough on this machine is exactly
the test that fails once a week on a loaded CI box.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from voicekey.session import Session

TIMEOUT = 2.0


def wait_for(predicate, timeout: float = TIMEOUT) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def burst(length: int = 16) -> np.ndarray:
    return np.zeros(length, dtype=np.float32)


class Harness:
    """A Session plus the callbacks' recorded output."""

    def __init__(self) -> None:
        self.appended: list[str] = []
        self.snapshots: list = []
        self.session = Session(on_append=self.appended.append, on_change=self.snapshots.append)

    @property
    def state(self) -> str:
        return self.session.display.state

    def close(self) -> None:
        self.session.close()


@pytest.fixture
def h():
    harness = Harness()
    yield harness
    harness.close()


def test_a_burst_produces_one_append(h):
    h.session.set_transcribe(lambda samples: "hello")
    h.session.submit(burst())

    assert wait_for(lambda: h.appended == ["hello"])


def test_appends_follow_submission_order(h):
    # Decode time varies per burst so that any reordering would show up here.
    def transcribe(samples: np.ndarray) -> str:
        time.sleep(0.02 if len(samples) > 50 else 0.001)
        return f"burst-{len(samples)}"

    h.session.set_transcribe(transcribe)
    for length in (100, 10, 100, 10):
        h.session.submit(burst(length))

    assert wait_for(lambda: len(h.appended) == 4)
    assert h.appended == ["burst-100", "burst-10", "burst-100", "burst-10"]


def test_bursts_recorded_before_the_model_loads_are_not_lost(h):
    # The premise of the app: recording starts instantly, the model loads in
    # the background, and nothing spoken in between is dropped.
    h.session.submit(burst())
    h.session.submit(burst())
    assert h.appended == []

    h.session.set_transcribe(lambda samples: "late")
    assert wait_for(lambda: h.appended == ["late", "late"])


def test_pending_rises_on_submit_and_returns_to_zero(h):
    release = threading.Event()

    def transcribe(samples: np.ndarray) -> str:
        release.wait(TIMEOUT)
        return "done"

    h.session.set_transcribe(transcribe)
    h.session.submit(burst())
    h.session.submit(burst())

    assert h.session.display.pending == 2
    release.set()
    assert wait_for(lambda: h.session.display.pending == 0)


def test_submitting_while_not_recording_does_not_report_idle(h):
    """The status must never claim idle with work outstanding.

    This is the state the UI shows after you stop talking, so getting it wrong
    means the app looks finished while a decode is still running.
    """
    release = threading.Event()
    h.session.set_transcribe(lambda samples: release.wait(TIMEOUT) and "done")
    h.session.set_recording(False)

    h.session.submit(burst())

    assert h.session.display.state == "transcribing"
    release.set()


def test_a_failing_transcribe_does_not_kill_the_worker(h):
    calls: list[int] = []

    def transcribe(samples: np.ndarray) -> str:
        calls.append(len(samples))
        if len(calls) == 1:
            raise RuntimeError("decode exploded")
        return "recovered"

    h.session.set_transcribe(transcribe)
    h.session.submit(burst())

    assert wait_for(lambda: h.session.display.state == "error")
    assert "decode exploded" in h.session.display.message

    h.session.submit(burst())
    assert wait_for(lambda: h.appended == ["recovered"])


def test_a_later_success_clears_the_error_message(h):
    calls: list[int] = []

    def transcribe(samples: np.ndarray) -> str:
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("boom")
        return "fine"

    h.session.set_transcribe(transcribe)
    h.session.submit(burst())
    assert wait_for(lambda: h.session.display.message != "")

    h.session.submit(burst())
    assert wait_for(lambda: h.appended == ["fine"])
    assert h.session.display.message == ""
    assert h.session.display.state != "error"


def test_silence_is_not_appended_but_still_completes(h):
    # An empty string is what the silence gate returns (voicekey.vad). It must
    # not land in the transcript as a blank, but it must still clear pending.
    results = iter(["", "spoken"])
    h.session.set_transcribe(lambda samples: next(results))

    h.session.submit(burst())
    h.session.submit(burst())

    assert wait_for(lambda: h.appended == ["spoken"])
    assert wait_for(lambda: h.session.display.pending == 0)


def test_recording_and_level_reach_the_snapshot(h):
    h.session.set_transcribe(lambda samples: "")
    h.session.set_recording(True)
    assert h.session.display.state == "recording"

    h.session.set_level(0.5)
    assert h.session.display.level == 0.5

    h.session.set_recording(False)
    assert h.session.display.state == "idle"


def test_an_unchanged_level_does_not_notify(h):
    h.session.set_level(0.25)
    before = len(h.snapshots)
    h.session.set_level(0.25)
    assert len(h.snapshots) == before


def test_close_returns_promptly_when_the_model_never_loaded():
    """Quitting during model load, with audio already recorded.

    The worker parks waiting for a model that will never arrive. If shutdown
    cannot interrupt that wait, close() blocks and the process never exits --
    for a GUI app, a hang the user has to force-kill.
    """
    session = Session(on_append=lambda text: None, on_change=lambda display: None)
    session.submit(burst())

    started = time.monotonic()
    session.close()
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, f"close() took {elapsed:.2f}s"
    assert not session._worker.is_alive()


def test_close_is_idempotent(h):
    h.session.set_transcribe(lambda samples: "x")
    h.session.close()
    h.session.close()
    assert not h.session._worker.is_alive()
