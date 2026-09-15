"""Log-mel spectrogram, matching OpenAI Whisper's front-end exactly.

Whisper is unforgiving here. The encoder was trained against one specific
preprocessing pipeline, and a subtly wrong filterbank or normalization does not
raise -- it produces fluent, confident nonsense. So this mirrors
`whisper.audio.log_mel_spectrogram` step for step, including the details that
look incidental:

* the STFT is centered, i.e. the signal is reflect-padded by `N_FFT // 2`
* the window is a *periodic* Hann (`torch.hann_window`'s default), which is
  `np.hanning(N + 1)[:-1]`, not `np.hanning(N)`
* the final STFT frame is dropped, which is what makes 30s land on exactly
  3000 frames
* the mel filterbank is librosa's Slaney-normalized bank (`htk=False`), not the
  HTK formula

There is no torch/librosa dependency -- the whole front-end is ~60 lines of
numpy, and it is the only real DSP in the project.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16_000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_SECONDS = 30
N_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS  # 480_000
N_FRAMES = N_SAMPLES // HOP_LENGTH  # 3_000

# Slaney mel scale: linear below 1kHz, logarithmic above.
_F_SP = 200.0 / 3.0
_MIN_LOG_HZ = 1000.0
_MIN_LOG_MEL = _MIN_LOG_HZ / _F_SP
_LOGSTEP = np.log(6.4) / 27.0


def _hz_to_mel(freq: np.ndarray) -> np.ndarray:
    freq = np.asarray(freq, dtype=np.float64)
    mels = freq / _F_SP
    log_region = freq >= _MIN_LOG_HZ
    if np.any(log_region):
        mels[log_region] = _MIN_LOG_MEL + np.log(freq[log_region] / _MIN_LOG_HZ) / _LOGSTEP
    return mels


def _mel_to_hz(mels: np.ndarray) -> np.ndarray:
    mels = np.asarray(mels, dtype=np.float64)
    freqs = mels * _F_SP
    log_region = mels >= _MIN_LOG_MEL
    if np.any(log_region):
        freqs[log_region] = _MIN_LOG_HZ * np.exp(_LOGSTEP * (mels[log_region] - _MIN_LOG_MEL))
    return freqs


def mel_filterbank(n_mels: int = 128, sr: int = SAMPLE_RATE, n_fft: int = N_FFT) -> np.ndarray:
    """librosa.filters.mel(htk=False, norm='slaney') -> (n_mels, n_fft//2 + 1)."""
    fft_freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    # n_mels + 2 edges: each filter spans three consecutive points.
    mel_pts = np.linspace(
        _hz_to_mel(np.array([0.0]))[0], _hz_to_mel(np.array([sr / 2]))[0], n_mels + 2
    )
    freq_pts = _mel_to_hz(mel_pts)

    fdiff = np.diff(freq_pts)
    ramps = freq_pts[:, None] - fft_freqs[None, :]

    weights = np.zeros((n_mels, len(fft_freqs)), dtype=np.float64)
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0.0, np.minimum(lower, upper))

    # Slaney normalization: equal *area* per filter, not equal peak.
    enorm = 2.0 / (freq_pts[2 : n_mels + 2] - freq_pts[:n_mels])
    weights *= enorm[:, None]
    return weights


def _stft_power(audio: np.ndarray) -> np.ndarray:
    """Centered STFT magnitude-squared -> (n_fft//2 + 1, n_frames)."""
    padded = np.pad(audio, N_FFT // 2, mode="reflect")
    window = np.hanning(N_FFT + 1)[:-1]  # periodic, matches torch.hann_window

    frames = np.lib.stride_tricks.sliding_window_view(padded, N_FFT)[::HOP_LENGTH]
    spec = np.fft.rfft(frames * window, axis=-1)
    # Whisper drops the final frame; this is what makes 30s == 3000 frames.
    return (np.abs(spec) ** 2).T[:, :-1]


def log_mel(audio: np.ndarray, n_mels: int = 128, filters: np.ndarray | None = None) -> np.ndarray:
    """16kHz mono f32 -> (n_mels, n_frames) float32, Whisper-normalized."""
    if filters is None:
        filters = mel_filterbank(n_mels)

    mel_spec = filters @ _stft_power(np.asarray(audio, dtype=np.float64))

    log_spec = np.log10(np.maximum(mel_spec, 1e-10))
    # Clamp the dynamic range to 80dB below the peak, then map to roughly [-1, 1].
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec.astype(np.float32)


def pad_or_trim(audio: np.ndarray, length: int = N_SAMPLES) -> np.ndarray:
    """Whisper's encoder is fixed-length: exactly 30 seconds, zero-padded."""
    if len(audio) > length:
        return audio[:length]
    if len(audio) < length:
        return np.pad(audio, (0, length - len(audio)))
    return audio
