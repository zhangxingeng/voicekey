"""Microphone level for the visual cue.

Purpose: turn a block of audio samples into a single 0..1 number the UI can
draw as a bar. This exists so a dead or muted microphone is visible *before*
the user finishes speaking, which is the failure the cue is there to catch.

Must be a log (dB) scale, not raw RMS: speech sits around 0.01-0.1 RMS, so a
linear bar looks nearly dead during normal speech and the cue is useless.
"""

from __future__ import annotations

import numpy as np

# Below this, treat as silence and return 0.0. Chosen so room tone (measured
# ~0.0086 RMS, see voicekey.vad) sits low on the bar but is not pinned to zero
# -- the user should be able to tell "mic is live but quiet" from "mic is dead".
FLOOR_DB = -60.0

# Above this, the bar is full. Speech peaks measured ~0.14 RMS (-17 dB).
CEILING_DB = -12.0


def level(samples: np.ndarray) -> float:
    """Map a block of samples to 0..1 for display.

    Returns 0.0 for an empty block or for digital silence. The scale is dB,
    clamped between FLOOR_DB and CEILING_DB and normalised to 0..1.
    """
    samples = np.asarray(samples, dtype=np.float64)
    if len(samples) == 0:
        return 0.0

    rms = float(np.sqrt((samples**2).mean()))

    # Guard log10 against zero. Silence must return 0.0, not -inf.
    if rms == 0.0:
        return 0.0

    db = 20.0 * np.log10(rms)

    # Clamp to display range, then map linearly to 0..1.
    clamped = max(FLOOR_DB, min(CEILING_DB, db))
    normalized = (clamped - FLOOR_DB) / (CEILING_DB - FLOOR_DB)

    return float(normalized)
