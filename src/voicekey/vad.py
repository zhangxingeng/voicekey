"""Is there actually speech in this audio?

Whisper hallucinates confidently on silence. Measured on this model, digital
silence decodes to "you" and room tone to "." -- so without a gate, triggering
the hotkey and then saying nothing pastes invented text.

The model's own <|nospeech|> head would be the principled signal, but this
export does not produce one: P(<|nospeech|>) measures 0.000000 for silence and
speech alike. So the check has to come from the audio.

What separates them, measured on a real microphone:

    room tone   rms 0.0086   crest 1.1     (steady, flat)
    speech      rms 0.1421   crest 5.5     (peaky, intermittent)

An absolute RMS threshold would encode this microphone's gain and break on any
other one. Instead the noise floor is estimated from the recording itself and
speech is defined relative to it, which is gain-independent: what matters is
that some frames stand well above the quiet ones, not how loud anything is in
absolute terms.
"""

from __future__ import annotations

import numpy as np

FRAME = 480  # 30ms at 16kHz

# A frame counts as speech at this multiple of the estimated noise floor.
# Room tone sits ~1.1x its own floor; speech peaks 15x or more above it.
FLOOR_MULTIPLE = 4.0

# Absolute floor, to reject digital silence and near-silence where the
# noise-floor estimate is meaningless (ratios explode when dividing by ~0).
ABS_FLOOR = 0.003

# Fraction of frames that must qualify. Low enough for a single short word in a
# long recording, high enough that one click or keyboard tap does not pass.
MIN_SPEECH_FRACTION = 0.05


def frame_rms(samples: np.ndarray, frame: int = FRAME) -> np.ndarray:
    """Per-frame RMS. Trailing partial frame is dropped."""
    usable = len(samples) - (len(samples) % frame)
    if usable < frame:
        return np.zeros(0, dtype=np.float64)
    frames = np.asarray(samples[:usable], dtype=np.float64).reshape(-1, frame)
    return np.sqrt((frames**2).mean(axis=1))


def speech_fraction(samples: np.ndarray) -> float:
    """Fraction of frames that stand out above the recording's own noise floor."""
    rms = frame_rms(samples)
    if len(rms) == 0:
        return 0.0
    # 10th percentile approximates "the quiet parts" without assuming any
    # particular amount of silence is present.
    floor = float(np.percentile(rms, 10))
    threshold = max(floor * FLOOR_MULTIPLE, ABS_FLOOR)
    return float((rms > threshold).mean())


def has_speech(samples: np.ndarray) -> bool:
    return speech_fraction(samples) >= MIN_SPEECH_FRACTION
