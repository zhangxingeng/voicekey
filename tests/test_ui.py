"""UI checks that need no display.

Constructing a Tk window requires an X/Wayland connection, which CI does not
have, so these cover the parts that are plain data: every state the app can
enter must have a cue, or `set_state` raises KeyError mid-transcription.
"""

from typing import get_args

from voicekey.ui import _CUES, _HINT, State


def test_every_state_has_a_cue():
    assert set(get_args(State)) == set(_CUES)


def test_cues_have_a_colour_and_a_caption():
    for colour, caption in _CUES.values():
        assert colour.startswith("#") and len(colour) == 7
        assert caption


def test_recording_and_idle_are_visually_distinct():
    # The dot is the primary cue; these two must never look the same.
    assert _CUES["recording"][0] != _CUES["idle"][0]


def test_hint_documents_the_only_three_controls():
    # There are no buttons, so the hint line is the entire discoverable UI.
    for key in ("Space", "Ctrl+C", "Esc"):
        assert key in _HINT
