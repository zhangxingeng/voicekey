"""Tests for model download and extraction.

The central promise of this module is that a partial model can never look like
a complete one -- `is_present` can only check that filenames exist, so anything
that leaves a truncated file under a real name poisons every later run and the
user has no way to recover short of deleting the directory by hand. Most of
these tests exist to hold that line.
"""

from __future__ import annotations

import hashlib
import tarfile
from pathlib import Path
from typing import NamedTuple

import pytest

from voicekey import models
from voicekey.models import MEMBERS, MODEL_SUBDIR, download, ensure_model, is_present


class TarFixture(NamedTuple):
    path: Path
    sha256: str


@pytest.fixture
def tar_fixture(tmp_path: Path) -> TarFixture:
    """A real .tar.bz2 shaped like the release asset, with one member that
    must not be extracted (the real archive carries a test_wavs/ directory)."""
    archive_path = tmp_path / "test-model.tar.bz2"
    content = tmp_path / "content" / MODEL_SUBDIR
    content.mkdir(parents=True)

    for member in MEMBERS:
        (content / member).write_bytes(b"dummy content for " + member.encode())
    (content / "extra-file-ignored.txt").write_bytes(b"should not be extracted")

    with tarfile.open(archive_path, "w:bz2") as tar:
        tar.add(content, arcname=MODEL_SUBDIR, filter=lambda x: x)

    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    return TarFixture(path=archive_path, sha256=digest)


@pytest.fixture
def models_root(tmp_path: Path) -> Path:
    root = tmp_path / "models"
    root.mkdir()
    return root


@pytest.fixture
def good_fetch(tar_fixture: TarFixture, monkeypatch):
    """A fetch that delivers the fixture archive, with the checksum to match."""
    monkeypatch.setattr(models, "MODEL_SHA256", tar_fixture.sha256)

    def fetch(url: str, dest: Path, on_progress) -> None:
        dest.write_bytes(tar_fixture.path.read_bytes())

    return fetch


class _MockResponse:
    """Enough of an http response for `download` to stream."""

    def __init__(self, content: bytes, *, send_length: bool = True) -> None:
        self.content = content
        self.headers = {"Content-Length": str(len(content))} if send_length else {}
        self._pos = 0

    def read(self, size: int = -1) -> bytes:
        if size == -1:
            size = len(self.content)
        chunk = self.content[self._pos : self._pos + size]
        self._pos += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


@pytest.fixture
def serve(monkeypatch):
    """Point urlopen at in-memory bytes."""

    def _serve(content: bytes, *, send_length: bool = True):
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda url, timeout=None: _MockResponse(content, send_length=send_length),
        )

    return _serve


class TestIsPresent:
    def test_false_on_empty_dir(self, models_root: Path) -> None:
        assert not is_present(models_root)

    def test_false_on_partial_members(self, models_root: Path) -> None:
        model_dir = models_root / MODEL_SUBDIR
        model_dir.mkdir()
        (model_dir / MEMBERS[0]).write_text("dummy")
        assert not is_present(models_root)

    def test_true_when_all_members_exist(self, models_root: Path) -> None:
        model_dir = models_root / MODEL_SUBDIR
        model_dir.mkdir()
        for member in MEMBERS:
            (model_dir / member).write_text("dummy")
        assert is_present(models_root)

    def test_false_when_member_is_a_directory(self, models_root: Path) -> None:
        model_dir = models_root / MODEL_SUBDIR
        model_dir.mkdir()
        for member in MEMBERS:
            (model_dir / member).mkdir()
        assert not is_present(models_root)


class TestDownload:
    def test_writes_content_to_dest(self, tmp_path: Path, serve) -> None:
        serve(b"test content")
        dest = tmp_path / "downloaded.bin"
        download("http://example.com/file", dest)
        assert dest.read_bytes() == b"test content"

    def test_reports_progress_ending_at_total(self, tmp_path: Path, serve) -> None:
        serve(b"x" * 100)
        calls: list[tuple[int, int]] = []
        download(
            "http://example.com/file",
            tmp_path / "f.bin",
            on_progress=lambda done, total: calls.append((done, total)),
        )
        assert calls[-1] == (100, 100)

    def test_tolerates_no_progress_callback(self, tmp_path: Path, serve) -> None:
        serve(b"test")
        dest = tmp_path / "downloaded.bin"
        download("http://example.com/file", dest, on_progress=None)
        assert dest.read_bytes() == b"test"

    def test_missing_content_length_reports_zero_total(self, tmp_path: Path, serve) -> None:
        # GitHub's asset redirect does not always send a length, and a progress
        # bar that divides by it would blow up rather than degrade.
        serve(b"y" * 10, send_length=False)
        calls: list[tuple[int, int]] = []
        download(
            "http://example.com/file",
            tmp_path / "f.bin",
            on_progress=lambda d, t: calls.append((d, t)),
        )
        assert calls
        assert all(total == 0 for _done, total in calls)


class TestEnsureModel:
    def test_extracts_only_the_members_it_needs(self, models_root: Path, good_fetch) -> None:
        result = ensure_model(models_root, fetch=good_fetch)

        assert result == models_root / MODEL_SUBDIR
        assert is_present(models_root)
        assert not (result / "extra-file-ignored.txt").exists()

    def test_is_a_no_op_when_already_present(self, models_root: Path) -> None:
        model_dir = models_root / MODEL_SUBDIR
        model_dir.mkdir()
        for member in MEMBERS:
            (model_dir / member).write_text("existing")

        called = False

        def fetch(url: str, dest: Path, on_progress) -> None:
            nonlocal called
            called = True

        assert ensure_model(models_root, fetch=fetch) == model_dir
        assert not called

    def test_checksum_mismatch_raises_and_leaves_nothing(
        self, models_root: Path, tar_fixture: TarFixture, monkeypatch
    ) -> None:
        monkeypatch.setattr(models, "MODEL_SHA256", "0" * 64)

        def fetch(url: str, dest: Path, on_progress) -> None:
            dest.write_bytes(tar_fixture.path.read_bytes())

        with pytest.raises(ValueError, match="checksum mismatch"):
            ensure_model(models_root, fetch=fetch)

        assert not is_present(models_root)
        assert list(models_root.iterdir()) == []

    def test_forwards_progress_to_fetch(self, models_root: Path, tar_fixture, monkeypatch) -> None:
        monkeypatch.setattr(models, "MODEL_SHA256", tar_fixture.sha256)
        seen: list[tuple[int, int]] = []

        def fetch(url: str, dest: Path, on_progress) -> None:
            dest.write_bytes(tar_fixture.path.read_bytes())
            on_progress(1, 2)

        ensure_model(models_root, fetch=fetch, on_progress=lambda d, t: seen.append((d, t)))
        assert seen == [(1, 2)]

    def test_stages_beside_the_destination_not_in_tmp(
        self, models_root: Path, tar_fixture, monkeypatch
    ) -> None:
        """The staging dir must share a filesystem with the destination.

        This is the whole reason the final step can be a rename. If staging
        moved back to /tmp, the rename would silently become a 1GB copy into
        the live path and a kill mid-copy would leave a truncated model that
        is_present() reports as valid.
        """
        monkeypatch.setattr(models, "MODEL_SHA256", tar_fixture.sha256)
        staged_under: list[Path] = []

        def fetch(url: str, dest: Path, on_progress) -> None:
            staged_under.append(dest.parent)
            dest.write_bytes(tar_fixture.path.read_bytes())

        ensure_model(models_root, fetch=fetch)

        assert staged_under[0].parent == models_root

    def test_leaves_no_staging_directory_behind(self, models_root: Path, good_fetch) -> None:
        ensure_model(models_root, fetch=good_fetch)
        assert [p.name for p in models_root.iterdir()] == [MODEL_SUBDIR]

    def test_a_crash_before_the_rename_leaves_no_model(
        self, models_root: Path, good_fetch, monkeypatch
    ) -> None:
        """Everything is staged, then one rename publishes it. Dying before
        that rename must leave the destination absent, not half-written."""

        def boom(src, dst):
            raise OSError("killed before publish")

        monkeypatch.setattr(models.os, "rename", boom)

        with pytest.raises(OSError, match="killed before publish"):
            ensure_model(models_root, fetch=good_fetch)

        assert not is_present(models_root)
        assert list(models_root.iterdir()) == []

    def test_replaces_a_partial_model_from_an_earlier_run(
        self, models_root: Path, good_fetch
    ) -> None:
        """A directory left by an interrupted run cannot be trusted or merged
        into -- it is replaced wholesale."""
        model_dir = models_root / MODEL_SUBDIR
        model_dir.mkdir()
        (model_dir / MEMBERS[0]).write_text("truncated leftover")
        (model_dir / "stale.txt").write_text("from a previous release")

        ensure_model(models_root, fetch=good_fetch)

        assert is_present(models_root)
        assert not (model_dir / "stale.txt").exists()
        assert (model_dir / MEMBERS[0]).read_bytes() != b"truncated leftover"

    def test_missing_member_in_archive_is_reported(
        self, models_root: Path, tmp_path: Path, monkeypatch
    ) -> None:
        incomplete = tmp_path / "incomplete.tar.bz2"
        content = tmp_path / "partial" / MODEL_SUBDIR
        content.mkdir(parents=True)
        (content / MEMBERS[0]).write_bytes(b"only one member")
        with tarfile.open(incomplete, "w:bz2") as tar:
            tar.add(content, arcname=MODEL_SUBDIR, filter=lambda x: x)

        monkeypatch.setattr(
            models, "MODEL_SHA256", hashlib.sha256(incomplete.read_bytes()).hexdigest()
        )

        def fetch(url: str, dest: Path, on_progress) -> None:
            dest.write_bytes(incomplete.read_bytes())

        with pytest.raises(KeyError):
            ensure_model(models_root, fetch=fetch)

        assert not is_present(models_root)
