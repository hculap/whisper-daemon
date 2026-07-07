"""Tests for config Settings new fields + save/load round-trip."""

import importlib

import pytest

from whisper_daemon import config as config_mod
from whisper_daemon.config import Settings


def test_new_fields_defaults():
    s = Settings()
    assert s.capture_mic is True
    assert s.capture_tab is True
    assert s.live_captions is False
    assert s.screenshot_displays == "all"


def _reload_config_with_dir(tmp_path, monkeypatch):
    """Point config at a temp config dir/file and reload module-level paths."""
    monkeypatch.setattr(config_mod, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config_mod, "CONFIG_FILE", tmp_path / "config.toml")


def test_save_load_roundtrip_preserves_new_fields(tmp_path, monkeypatch):
    _reload_config_with_dir(tmp_path, monkeypatch)

    original = Settings(
        capture_mic=False,
        capture_tab=False,
        live_captions=True,
        screenshot_displays="primary",
        recording_formats=["txt", "srt"],
        diarize=True,
        diarize_mode="batch",
    )
    config_mod.save_settings(original)
    loaded = config_mod.load_settings()

    assert loaded.capture_mic is False
    assert loaded.capture_tab is False
    assert loaded.live_captions is True
    assert loaded.screenshot_displays == "primary"
    assert loaded.recording_formats == ["txt", "srt"]
    assert loaded.diarize is True
    assert loaded.diarize_mode == "batch"


def test_load_defaults_when_no_file(tmp_path, monkeypatch):
    _reload_config_with_dir(tmp_path, monkeypatch)
    loaded = config_mod.load_settings()
    assert loaded.capture_mic is True
    assert loaded.capture_tab is True
    assert loaded.live_captions is False
    assert loaded.screenshot_displays == "all"


def test_screenshot_displays_validator_invalid(tmp_path, monkeypatch):
    _reload_config_with_dir(tmp_path, monkeypatch)
    # write a config with an invalid screenshot_displays value
    (tmp_path / "config.toml").write_text(
        '[recording]\nscreenshot_displays = "left"\n', encoding="utf-8"
    )
    loaded = config_mod.load_settings()
    assert loaded.screenshot_displays == "all"


def test_screenshot_displays_validator_valid(tmp_path, monkeypatch):
    _reload_config_with_dir(tmp_path, monkeypatch)
    (tmp_path / "config.toml").write_text(
        '[recording]\nscreenshot_displays = "primary"\n', encoding="utf-8"
    )
    loaded = config_mod.load_settings()
    assert loaded.screenshot_displays == "primary"
