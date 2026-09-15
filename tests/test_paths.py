from pathlib import Path

from voicekey.paths import data_dir, models_dir


def test_env_override_wins(monkeypatch):
    monkeypatch.setenv("VOICEKEY_DATA_DIR", "/tmp/vk-test")
    assert data_dir() == Path("/tmp/vk-test")
    assert models_dir() == Path("/tmp/vk-test/models")


def test_blank_override_is_ignored(monkeypatch):
    monkeypatch.setenv("VOICEKEY_DATA_DIR", "   ")
    assert data_dir() != Path("   ")
    assert "voicekey" in str(data_dir()).lower()


def test_models_dir_is_under_data_dir(monkeypatch):
    monkeypatch.delenv("VOICEKEY_DATA_DIR", raising=False)
    assert models_dir().parent == data_dir()
