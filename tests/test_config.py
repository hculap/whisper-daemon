"""Tests for config Settings new fields + save/load round-trip."""

import importlib
import tomllib

import pytest

from whisper_daemon import config as config_mod
from whisper_daemon.config import Settings, _toml_str


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


# --- F12: _toml_str escapes control chars so output always re-parses -------


@pytest.mark.parametrize(
    "value",
    [
        "plain",
        "with\ttab",
        "form\x0cfeed",
        "new\nline",
        "carriage\rreturn",
        'has"quote',
        "has\\backslash",
        "bell\x07and\x7fdel",
        "~/Meetings/Zoom – Team",
    ],
)
def test_toml_str_output_reparses(value):
    # Wrap in a key/value line and confirm tomllib parses back the exact value,
    # so even a control char that slipped past validation cannot corrupt config.
    doc = f"dir = {_toml_str(value)}\n"
    parsed = tomllib.loads(doc)
    assert parsed["dir"] == value


def test_toml_str_control_char_is_escaped_not_literal():
    out = _toml_str("a\tb\x0cc")
    # No raw control chars remain in the serialized form.
    assert "\t" not in out
    assert "\x0c" not in out
    assert out.startswith('"') and out.endswith('"')


def test_save_load_roundtrip_with_control_char_dir(tmp_path, monkeypatch):
    # A recording_dir containing a control char (e.g. if it ever bypassed the
    # patch validator) must still produce a config that loads without error.
    _reload_config_with_dir(tmp_path, monkeypatch)
    config_mod.save_settings(Settings(recording_dir="~/Meet\ting\x0c"))
    loaded = config_mod.load_settings()
    assert loaded.recording_dir == "~/Meet\ting\x0c"


def test_toml_str_drops_lone_surrogate():
    # A lone surrogate is not utf-8 encodable and a bare \uD800 escape is not a
    # valid TOML scalar — _toml_str must drop it so the output stays writable.
    out = _toml_str("dir\ud800name")
    assert "\ud800" not in out
    out.encode("utf-8")  # would raise UnicodeEncodeError if surrogate leaked
    assert tomllib.loads(f"dir = {out}\n")["dir"] == "dirname"


def test_save_with_surrogate_does_not_truncate_existing_config(tmp_path, monkeypatch):
    # A Settings carrying a lone surrogate must not truncate an existing
    # config.toml to zero bytes (F12: no value can corrupt config on save).
    _reload_config_with_dir(tmp_path, monkeypatch)
    config_mod.save_settings(Settings(recording_dir="~/Meetings"))
    before = (tmp_path / "config.toml").read_text(encoding="utf-8")
    assert before  # file has real content

    config_mod.save_settings(Settings(recording_dir="~/Meet\ud800ings"))
    after = (tmp_path / "config.toml").read_text(encoding="utf-8")
    assert after  # not truncated to zero bytes
    loaded = config_mod.load_settings()
    assert loaded.recording_dir == "~/Meetings"  # surrogate dropped, dir intact


def test_screenshot_displays_validator_valid(tmp_path, monkeypatch):
    _reload_config_with_dir(tmp_path, monkeypatch)
    (tmp_path / "config.toml").write_text(
        '[recording]\nscreenshot_displays = "primary"\n', encoding="utf-8"
    )
    loaded = config_mod.load_settings()
    assert loaded.screenshot_displays == "primary"
