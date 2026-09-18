"""UI checks that need no display.

Constructing a Tk window requires an X/Wayland connection, which CI does not
have, so these cover the parts that are plain data. That is a real limit worth
naming: the layout, the meter drawing and the append-preserves-caret behaviour
are only ever exercised by hand.
"""

from __future__ import annotations

from typing import get_args

from voicekey.display import State
from voicekey.ui import _CUES, _DEAD_LEVEL, _HINT, _TOGGLE_SEQUENCES


def test_every_state_has_a_cue():
    # set_state raises KeyError mid-transcription if one is missing.
    assert set(get_args(State)) == set(_CUES)


def test_cues_have_a_colour_and_a_caption():
    for colour, caption in _CUES.values():
        assert colour.startswith("#") and len(colour) == 7
        assert caption


def test_recording_and_idle_are_visually_distinct():
    # The dot is the primary cue; these two must never look the same.
    assert _CUES["recording"][0] != _CUES["idle"][0]


def test_hint_names_the_controls():
    for key in ("Super+Shift+D", "Ctrl+C", "Esc"):
        assert key in _HINT


def test_dead_level_threshold_is_below_room_tone():
    """A live-but-quiet mic must not be reported as dead.

    Room tone measures ~0.0086 RMS, which voicekey.meter maps to roughly 0.39
    on the bar (see tests/test_meter.py). The dead threshold has to sit well
    under that, or every quiet room looks like a broken microphone.
    """
    assert _DEAD_LEVEL < 0.39


def test_toggle_sequences_use_tk_modifier_spelling():
    """Tk has no "Super" modifier -- the Super key is Mod4.

    Binding an unrecognised sequence raises TclError at bind time, and this
    one runs in the constructor, so getting it wrong stopped the window from
    opening at all rather than just losing a shortcut.
    """
    for sequence in _TOGGLE_SEQUENCES:
        assert "Super" not in sequence
        assert sequence.startswith("<Mod4-")


def test_both_letter_cases_are_bound():
    # With Shift held, X11 reports the keysym uppercase on some layouts and
    # lowercase on others; binding only one silently works on one machine.
    assert {s[-2] for s in _TOGGLE_SEQUENCES} == {"D", "d"}
