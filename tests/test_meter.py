"""Microphone level meter: turning audio into a 0..1 bar value.

The meter should:
  - Show silence and dead mics as zero
  - Show room tone (mic live but quiet) low but visible
  - Show speech-level signals in the mid-to-high range
  - Never saturate; always stay within [0, 1]

Tests use deterministic signals: constant-amplitude arrays and sine waves with
fixed shape. No randomness without a fixed seed.
"""

import numpy as np

from voicekey import meter

RATE = 16_000


def _constant_amplitude_rms(amplitude):
    """Constant value has RMS equal to its absolute value."""
    return np.full(RATE, amplitude, dtype=np.float64)


def _sine_wave(frequency, duration_s, amplitude):
    """Sine wave with given frequency, duration, and peak amplitude.

    Peak amplitude is the amplitude parameter. RMS of a sine is amplitude / sqrt(2).
    """
    t = np.linspace(0, duration_s, int(RATE * duration_s), endpoint=False)
    return (np.sin(2 * np.pi * frequency * t) * amplitude).astype(np.float64)


def test_empty_returns_zero():
    """Empty block has no level."""
    assert meter.level(np.array([], dtype=np.float32)) == 0.0


def test_silence_returns_zero():
    """All-zero samples (digital silence) must return 0.0."""
    assert meter.level(np.zeros(RATE, dtype=np.float32)) == 0.0
    # Also test numpy array already in float64.
    assert meter.level(np.zeros(RATE, dtype=np.float64)) == 0.0


def test_silence_list_input():
    """Function must accept lists and convert them to arrays."""
    assert meter.level([0.0, 0.0, 0.0]) == 0.0


def test_room_tone_sits_low_but_visible():
    """Room tone at 0.0086 RMS sits low on the bar (between 0 and mid-range).

    This is the "mic is live but quiet" case the UI needs to show.
    Expected dB: 20 * log10(0.0086) = -41.3 dB.
    Normalized: (-41.3 - (-60)) / (-12 - (-60)) = 18.7 / 48 = 0.39.
    """
    room_tone = _constant_amplitude_rms(0.0086)
    level = meter.level(room_tone)
    assert 0.0 < level < 0.5
    # More precisely: should be around 0.39.
    assert 0.38 < level < 0.40


def test_speech_level_sits_mid_to_high():
    """Speech at 0.14 RMS sits in the mid-to-high range.

    Expected dB: 20 * log10(0.14) = -17.1 dB.
    Normalized: (-17.1 - (-60)) / (-12 - (-60)) = 42.9 / 48 = 0.89.
    """
    speech = _constant_amplitude_rms(0.14)
    level = meter.level(speech)
    assert 0.75 < level < 1.0
    # More precisely: should be around 0.89.
    assert 0.88 < level < 0.90


def test_loud_signal_near_full_scale():
    """A loud signal (amplitude 0.8) sits near or at full scale.

    Expected dB: 20 * log10(0.8) = -1.94 dB.
    This exceeds CEILING_DB = -12, so it clamps to 1.0.
    """
    loud = _constant_amplitude_rms(0.8)
    level = meter.level(loud)
    assert level == 1.0


def test_ceiling_clipping():
    """Signals above CEILING_DB clamp to 1.0."""
    # 20 * log10(0.26) = -11.67 dB, which is above CEILING_DB = -12.0 dB.
    just_above_ceiling = _constant_amplitude_rms(0.26)
    assert meter.level(just_above_ceiling) == 1.0

    # Definitely above CEILING_DB.
    definitely_loud = _constant_amplitude_rms(1.0)
    assert meter.level(definitely_loud) == 1.0


def test_floor_clipping():
    """Signals below FLOOR_DB clamp to 0.0.

    FLOOR_DB = -60.0, so anything at RMS < 20^(-60/20) = 0.001 should be ~0.
    """
    very_quiet = _constant_amplitude_rms(0.0001)
    level = meter.level(very_quiet)
    assert level == 0.0


def test_output_always_in_range():
    """For a range of input amplitudes, output stays within [0, 1]."""
    for amplitude in [0.0, 0.00001, 0.001, 0.01, 0.1, 0.2, 0.5, 1.0, 2.0]:
        signal = _constant_amplitude_rms(amplitude)
        level = meter.level(signal)
        assert 0.0 <= level <= 1.0


def test_returns_python_float():
    """Output must be a plain Python float, not a numpy scalar."""
    room_tone = _constant_amplitude_rms(0.0086)
    level = meter.level(room_tone)
    assert isinstance(level, float)
    assert not isinstance(level, np.floating)


def test_sine_wave_room_tone():
    """A 1 kHz sine wave at 0.0086 peak amplitude has RMS 0.0086/sqrt(2).

    This should sit low and visible, similar to the constant-amplitude case.
    """
    # Peak amplitude 0.0121 gives RMS = 0.0121/sqrt(2) = 0.0086.
    sine = _sine_wave(1000.0, 1.0, 0.0121)
    level = meter.level(sine)
    # Should be in the same ballpark as the constant room-tone case.
    assert 0.0 < level < 0.5


def test_sine_wave_speech():
    """A sine wave modeled to have speech-level RMS."""
    # Peak amplitude = 0.14 * sqrt(2) = 0.198.
    sine = _sine_wave(1000.0, 1.0, 0.198)
    level = meter.level(sine)
    # Should be in the mid-to-high range, similar to speech.
    assert 0.75 < level < 1.0


def test_short_samples():
    """Very short blocks must still produce valid output."""
    short_silence = np.zeros(10, dtype=np.float32)
    assert meter.level(short_silence) == 0.0

    short_room_tone = _constant_amplitude_rms(0.0086)[:10]
    level = meter.level(short_room_tone)
    assert 0.0 < level < 0.5


def test_stereo_like_input_is_flattened():
    """If given a 2D-like array, it should compute RMS of all elements."""
    # A 2D array of silence.
    two_d = np.zeros((100, 100), dtype=np.float32)
    assert meter.level(two_d) == 0.0

    # A 2D array with a specific RMS across all elements.
    two_d_room_tone = np.full((100, 100), 0.0086, dtype=np.float32)
    level = meter.level(two_d_room_tone)
    assert 0.38 < level < 0.40


def test_level_is_independent_of_block_length():
    """A longer block of the same signal must read the same on the bar.

    The meter is called on whatever chunk the audio callback hands over, and
    those vary in size -- the bar must not jump because a buffer was bigger.
    """
    one_second = _constant_amplitude_rms(0.0086)
    hundred_seconds = np.tile(one_second, 100)
    assert meter.level(one_second) == meter.level(hundred_seconds)
