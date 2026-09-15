"""Invariants for the Whisper front-end.

A wrong mel spectrogram does not crash -- it makes the model hallucinate
fluently. These tests pin the properties that would silently drift: frame
count, filterbank normalization, and the output range Whisper expects.
"""

import numpy as np

from voicekey import mel as m


def test_thirty_seconds_is_exactly_3000_frames():
    # This is the contract with the encoder: n_audio_ctx=1500, and the encoder
    # downsamples by 2. Off-by-one here corrupts every transcription.
    audio = np.zeros(m.N_SAMPLES, dtype=np.float32)
    assert m.log_mel(audio).shape == (128, 3000)


def test_output_is_float32_for_onnx():
    assert m.log_mel(np.zeros(m.N_SAMPLES)).dtype == np.float32


def test_dynamic_range_is_clamped_to_two():
    # log_spec is floored at peak-8 then mapped by (x+4)/4, so the span can
    # never exceed 2.0 regardless of input.
    rng = np.random.default_rng(0)
    for audio in (
        rng.standard_normal(m.N_SAMPLES) * 0.1,
        np.zeros(m.N_SAMPLES),
        np.sin(np.arange(m.N_SAMPLES) * 0.01),
    ):
        spec = m.log_mel(audio)
        assert spec.max() - spec.min() <= 2.0 + 1e-5


def test_floor_is_reached_when_audio_has_quiet_stretches():
    # A loud burst followed by silence drives the quiet bins onto the clamp,
    # so the span hits its 2.0 ceiling exactly.
    audio = np.zeros(m.N_SAMPLES, dtype=np.float32)
    t = np.arange(m.SAMPLE_RATE) / m.SAMPLE_RATE
    audio[: m.SAMPLE_RATE] = np.sin(2 * np.pi * 440.0 * t)
    spec = m.log_mel(audio)
    assert np.isclose(spec.max() - spec.min(), 2.0, atol=1e-4)


def test_filterbank_shape_and_support():
    f = m.mel_filterbank(128)
    assert f.shape == (128, m.N_FFT // 2 + 1)
    assert (f >= 0).all(), "triangular filters are non-negative"
    assert (f.sum(axis=1) > 0).all(), "no empty filter"


def test_filters_are_ordered_low_to_high():
    f = m.mel_filterbank(128)
    centres = f.argmax(axis=1)
    assert (np.diff(centres) >= 0).all()


def test_slaney_normalization_equalizes_area_not_peak():
    # Slaney norm scales each filter by 2/bandwidth, so higher (wider) filters
    # have visibly lower peaks. HTK norm would leave peaks equal -- this test
    # is what catches using the wrong mel convention.
    f = m.mel_filterbank(128)
    assert f[100].max() < f[10].max()


def test_pure_tone_lands_in_one_region():
    t = np.arange(m.N_SAMPLES) / m.SAMPLE_RATE
    spec = m.log_mel(np.sin(2 * np.pi * 440.0 * t).astype(np.float32))
    # Energy should concentrate in the low mel bins for a 440Hz tone.
    profile = spec.mean(axis=1)
    assert profile[:40].max() > profile[80:].max()


def test_pad_or_trim():
    assert len(m.pad_or_trim(np.zeros(100))) == m.N_SAMPLES
    assert len(m.pad_or_trim(np.zeros(m.N_SAMPLES * 2))) == m.N_SAMPLES
    exact = np.zeros(m.N_SAMPLES)
    assert m.pad_or_trim(exact) is exact


def test_padding_preserves_the_original_samples():
    audio = np.linspace(-1, 1, 1000).astype(np.float32)
    padded = m.pad_or_trim(audio)
    assert np.array_equal(padded[:1000], audio)
    assert (padded[1000:] == 0).all()
