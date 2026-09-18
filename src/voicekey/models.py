"""First-run model download.

A port of salvage/download-whisper-model.sh, which cannot ship as-is: it needs
curl, sha256sum and `tar -xjf`, none of which Windows has. This is the file
that decides whether a released build works for anyone but the author.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
import tempfile
import urllib.request
from collections.abc import Callable
from pathlib import Path

# Pinned k2-fsa release asset. int8-quantised Whisper large-v3-turbo, ~540MB
# archive, ~1.0GB extracted. SenseVoice-Small (3x smaller, 5x faster) was tried
# first and rejected: it rendered "GitHub" as "GET UP" and "Kubernetes" as
# "CORNATTIE ENGINES".
MODEL_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-turbo.tar.bz2"
MODEL_SHA256 = "b11acbbcd660b44a8e0df33724feb5aaa709cf65668f2823d59f656312544f22"
MODEL_SUBDIR = "sherpa-onnx-whisper-turbo"

# The only members the engine loads. The archive also carries test_wavs/.
MEMBERS = ("turbo-encoder.int8.onnx", "turbo-decoder.int8.onnx", "turbo-tokens.txt")

# 1MB. The default 8KB means ~66,000 syscalls for a 540MB archive.
CHUNK = 1 << 20

# A stalled connection mid-download is common on flaky wifi, and without this
# urlopen blocks forever behind a progress bar that simply stops moving.
TIMEOUT_S = 30.0

# (bytes_done, bytes_total) -- total is 0 when the server sends no length.
ProgressFn = Callable[[int, int], None]


def is_present(models_root: Path) -> bool:
    """True when every member of MEMBERS already exists under models_root."""
    subdir = models_root / MODEL_SUBDIR
    return all((subdir / member).is_file() for member in MEMBERS)


def download(url: str, dest: Path, on_progress: ProgressFn | None = None) -> None:
    """Fetch `url` to `dest`, reporting progress. The default fetcher.

    Split out from `ensure_model` purely so tests can inject a fake and never
    touch the network.
    """
    with urllib.request.urlopen(url, timeout=TIMEOUT_S) as response:
        # Absent on some servers, and absent from GitHub's asset redirect in
        # particular, so the caller must treat 0 as "length unknown".
        total = int(response.headers.get("Content-Length", 0))
        done = 0
        with open(dest, "wb") as f:
            while chunk := response.read(CHUNK):
                f.write(chunk)
                done += len(chunk)
                if on_progress is not None:
                    on_progress(done, total)


def _sha256(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def ensure_model(
    models_root: Path,
    *,
    fetch: Callable[[str, Path, ProgressFn | None], None] = download,
    on_progress: ProgressFn | None = None,
) -> Path:
    """Return the model directory, downloading and extracting it if needed.

    Verifies sha256 before extracting and discards the archive on mismatch --
    a truncated 540MB download must never be left looking like a valid model.

    Everything happens in a staging directory that is renamed into place as a
    single step at the end, so the model directory either does not exist or is
    complete. Moving the members in one at a time was tried first and is not
    good enough: `shutil.move` silently degrades from `os.rename` to a
    streaming copy whenever the source and destination are on different
    filesystems (/tmp is usually tmpfs, the destination is under $HOME), and
    that copy writes straight into the final path. Killed mid-copy of the last
    member, it leaves three files where one is truncated -- and `is_present`,
    which can only check that names exist, calls that a valid model forever.
    """
    model_dir = models_root / MODEL_SUBDIR
    if is_present(models_root):
        return model_dir

    # Staging lives under models_root, not /tmp, so the final rename is a
    # same-filesystem metadata operation rather than a 1GB copy.
    models_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=models_root, prefix=".staging-"))
    try:
        archive = staging / "model.tar.bz2"
        fetch(MODEL_URL, archive, on_progress)

        actual = _sha256(archive)
        if actual != MODEL_SHA256:
            raise ValueError(
                f"model archive checksum mismatch: expected {MODEL_SHA256}, got {actual}"
            )

        # Only the three members, never the whole archive: it also carries a
        # test_wavs/ directory that would otherwise ship to every user.
        with tarfile.open(archive, "r:bz2") as tar:
            for member in MEMBERS:
                tar.extract(f"{MODEL_SUBDIR}/{member}", path=staging, filter="data")

        archive.unlink()

        # A previous interrupted run may have left a partial directory. It
        # cannot be trusted, and os.rename onto a non-empty directory fails.
        if model_dir.exists():
            shutil.rmtree(model_dir)
        os.rename(staging / MODEL_SUBDIR, model_dir)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    return model_dir
