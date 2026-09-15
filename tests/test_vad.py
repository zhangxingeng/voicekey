"""The silence gate.

Both directions matter and fail differently: a gate that is too eager drops
real speech (the user loses what they said), one that is too lax lets Whisper
paste invented text. The numbers here come from measurements on a real
microphone -- see voicekey.vad.
"""

import numpy as np

from voicekey import vad

RATE = 16_000
rng = np.random.default_rng(1234)


def _room_tone(seconds=3.0, level=0.0086):
    """Steady low-level noise: rms ~0.0086, crest ~1.1, as measured."""
    return (rng.standard_normal(int(RATE * seconds)) * level).astype(np.float32)


def _speech_like(seconds=3.0, level=0.14):
    """Intermittent bursts over a quiet floor -- the shape speech has."""
    n = int(RATE * seconds)
    sig = (rng.standard_normal(n) * 0.002).astype(np.float32)
    # Three bursts, ~40% of the duration in total.
    for start in (0.2, 1.2, 2.1):
        a = int(start * RATE)
        sig[a : a + int(0.4 * RATE)] += (rng.standard_normal(int(0.4 * RATE)) * level).astype(
            np.float32
        )
    return sig


def test_digital_silence_is_rejected():
    assert not vad.has_speech(np.zeros(RATE * 3, dtype=np.float32))


def test_room_tone_is_rejected():
    assert not vad.has_speech(_room_tone())


def test_loud_but_steady_noise_is_rejected():
    # A fan at high gain: plenty of energy, no speech structure. An absolute
    # RMS threshold would wrongly accept this.
    assert not vad.has_speech(_room_tone(level=0.12))


def test_speech_is_accepted():
    assert vad.has_speech(_speech_like())


def test_quiet_speech_is_still_accepted():
    # Low mic gain must not lose real speech -- this is why the threshold is
    # relative to the recording's own noise floor rather than absolute.
    assert vad.has_speech(_speech_like(level=0.02))


def test_single_click_does_not_pass():
    sig = np.zeros(RATE * 5, dtype=np.float32)
    sig[RATE] = 1.0
    assert not vad.has_speech(sig)


def test_one_short_word_in_a_long_recording_passes():
    sig = (rng.standard_normal(RATE * 10) * 0.002).astype(np.float32)
    sig[RATE : RATE + int(0.8 * RATE)] += (rng.standard_normal(int(0.8 * RATE)) * 0.15).astype(
        np.float32
    )
    assert vad.has_speech(sig)


def test_empty_and_tiny_inputs_are_safe():
    assert not vad.has_speech(np.zeros(0, dtype=np.float32))
    assert not vad.has_speech(np.zeros(10, dtype=np.float32))
    assert vad.speech_fraction(np.zeros(0, dtype=np.float32)) == 0.0


def test_frame_rms_drops_partial_trailing_frame():
    assert len(vad.frame_rms(np.zeros(vad.FRAME * 3 + 17))) == 3
