"""Where model files live, per platform.

Models are never packaged -- they are 1-2GB and get downloaded on first run.
platformdirs puts them in the right place on each OS:

    Linux    ~/.local/share/voicekey/models
    macOS    ~/Library/Application Support/voicekey/models
    Windows  %LOCALAPPDATA%\\voicekey\\models
"""

from __future__ import annotations

import os
from pathlib import Path

from platformdirs import user_data_path

APP_NAME = "voicekey"


def data_dir() -> Path:
    """App data root. `VOICEKEY_DATA_DIR` overrides it (tests, custom setups)."""
    override = os.environ.get("VOICEKEY_DATA_DIR", "").strip()
    if override:
        return Path(override)
    return user_data_path(APP_NAME, appauthor=False)


def models_dir() -> Path:
    return data_dir() / "models"
