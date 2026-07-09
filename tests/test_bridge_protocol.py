"""Tests for the pure bridge_protocol module (no heavy imports)."""

import dataclasses
import json

import pytest

from whisper_daemon import bridge_protocol as bp
from whisper_daemon.config import Settings

# --- is_allowed_origin ---------------------------------------------------


@pytest.mark.parametrize(
    "origin,expected",
    [
        ("chrome-extension://abcdefghijklmnop", True),
        ("chrome-extension://", True),
        ("https://meet.google.com", False),
        ("http://localhost:9876", False),
        ("moz-extension://abc", False),
        ("", False),
        (None, False),
    ],
)
def test_is_allowed_origin(origin, expected):
    assert bp.is_allowed_origin(origin) is expected


# --- settings_to_dict ----------------------------------------------------


def test_settings_to_dict_exact_keys():
    d = bp.settings_to_dict(Settings())
    assert set(d.keys()) == {
        "recording_device",
        "capture_mic",
        "capture_tab",
        "capture_screenshots",
        "screenshot_displays",
        "live_captions",
        "recording_language",
        "diarize",
        "diarize_mode",
        "recording_dir",
        "recording_formats",
    }


def test_settings_to_dict_default_values():
    d = bp.settings_to_dict(Settings())
    assert d["recording_device"] == ""
    assert d["capture_mic"] is True
    assert d["capture_tab"] is True
    assert d["capture_screenshots"] is False
    assert d["screenshot_displays"] == "all"
    assert d["live_captions"] is False
    assert d["diarize"] is False
    assert d["diarize_mode"] == "hybrid"
    assert d["recording_dir"] == "~/Desktop"
    assert d["recording_formats"] == ["txt"]


def test_settings_to_dict_copies_formats_list():
    s = Settings(recording_formats=["txt", "srt"])
    d = bp.settings_to_dict(s)
    d["recording_formats"].append("vtt")
    # original settings list must be unchanged
    assert s.recording_formats == ["txt", "srt"]


# --- build_settings_message ----------------------------------------------


def test_build_settings_message_shape():
    raw = bp.build_settings_message(Settings(), ["MacBook Micro", "BlackHole 2ch"])
    msg = json.loads(raw)
    assert msg["type"] == "settings"
    assert msg["devices"] == ["MacBook Micro", "BlackHole 2ch"]
    assert msg["settings"]["capture_mic"] is True
    assert msg["settings"]["diarize_mode"] == "hybrid"


# --- build_status_message ------------------------------------------------


def test_build_status_message_shape():
    raw = bp.build_status_message("recording", 12.5, 3)
    msg = json.loads(raw)
    assert msg == {
        "type": "status",
        "state": "recording",
        "elapsed": 12.5,
        "chunks": 3,
    }


# --- parse_client_message ------------------------------------------------


def test_parse_client_message_valid():
    assert bp.parse_client_message('{"type":"ping"}') == {"type": "ping"}


def test_parse_client_message_invalid_json():
    assert bp.parse_client_message("not json {") is None


def test_parse_client_message_non_dict():
    assert bp.parse_client_message("[1, 2, 3]") is None
    assert bp.parse_client_message('"just a string"') is None
    assert bp.parse_client_message("42") is None


# --- apply_settings_patch ------------------------------------------------


def test_apply_patch_returns_new_object_immutable():
    s = Settings()
    result = bp.apply_settings_patch(s, {"capture_mic": False})
    assert result is not s
    assert result.capture_mic is False
    # original unchanged
    assert s.capture_mic is True


def test_apply_patch_original_formats_list_unchanged():
    s = Settings(recording_formats=["txt"])
    result = bp.apply_settings_patch(s, {"recording_formats": ["srt", "vtt"]})
    assert s.recording_formats == ["txt"]
    assert result.recording_formats == ["srt", "vtt"]


def test_apply_patch_unknown_key_ignored():
    s = Settings()
    result = bp.apply_settings_patch(s, {"totally_unknown": "value", "diarize": True})
    assert not hasattr(result, "totally_unknown")
    assert result.diarize is True


def test_apply_patch_invalid_format_dropped():
    s = Settings(recording_formats=["txt"])
    result = bp.apply_settings_patch(s, {"recording_formats": ["srt", "bogus", "vtt"]})
    assert result.recording_formats == ["srt", "vtt"]


def test_apply_patch_empty_formats_keeps_old():
    s = Settings(recording_formats=["txt", "srt"])
    result = bp.apply_settings_patch(s, {"recording_formats": ["bogus", "nope"]})
    assert result.recording_formats == ["txt", "srt"]
    result2 = bp.apply_settings_patch(s, {"recording_formats": []})
    assert result2.recording_formats == ["txt", "srt"]


def test_apply_patch_screenshot_displays_validation():
    s = Settings()
    assert bp.apply_settings_patch(s, {"screenshot_displays": "primary"}).screenshot_displays == "primary"
    assert bp.apply_settings_patch(s, {"screenshot_displays": "all"}).screenshot_displays == "all"
    # invalid value rejected -> keep old ("all")
    assert bp.apply_settings_patch(s, {"screenshot_displays": "left"}).screenshot_displays == "all"


def test_apply_patch_diarize_mode_validation():
    s = Settings()
    for mode in ("batch", "realtime", "hybrid"):
        assert bp.apply_settings_patch(s, {"diarize_mode": mode}).diarize_mode == mode
    # invalid rejected -> keep old
    assert bp.apply_settings_patch(s, {"diarize_mode": "nonsense"}).diarize_mode == "hybrid"


def test_apply_patch_partial():
    s = Settings(capture_mic=True, capture_tab=True, live_captions=False)
    result = bp.apply_settings_patch(s, {"live_captions": True})
    assert result.live_captions is True
    assert result.capture_mic is True
    assert result.capture_tab is True


def test_apply_patch_bool_coercion_strict():
    s = Settings()
    # non-bool values for bool fields are rejected (keep old)
    assert bp.apply_settings_patch(s, {"capture_mic": 1}).capture_mic is True
    assert bp.apply_settings_patch(s, {"capture_mic": "true"}).capture_mic is True
    assert bp.apply_settings_patch(s, {"capture_mic": 0}).capture_mic is True
    # a real bool is accepted
    assert bp.apply_settings_patch(s, {"capture_mic": False}).capture_mic is False


def test_apply_patch_string_type_enforced():
    s = Settings()
    # non-str for a string field rejected
    assert bp.apply_settings_patch(s, {"recording_device": 123}).recording_device == ""
    assert bp.apply_settings_patch(s, {"recording_device": "BlackHole 2ch"}).recording_device == "BlackHole 2ch"
    assert bp.apply_settings_patch(s, {"recording_dir": "~/Meetings"}).recording_dir == "~/Meetings"


def test_bridge_protocol_has_no_heavy_imports():
    import subprocess
    import sys

    # Import bridge_protocol in a fresh interpreter (the shared pytest process
    # loads Quartz via the screen_capture tests, so sys.modules here is dirty).
    code = (
        "import sys, whisper_daemon.bridge_protocol\n"
        "heavy = [m for m in ('mlx', 'mlx_whisper', 'torch', 'objc', 'Quartz')"
        " if m in sys.modules]\n"
        "print(','.join(heavy))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "", f"heavy modules imported: {out.stdout.strip()}"


# --- F12: control-char rejection at the patch boundary -------------------


@pytest.mark.parametrize(
    "bad",
    ["with\ttab", "form\x0cfeed", "bell\x07", "del\x7f", "vtab\x0b"],
)
def test_apply_patch_rejects_control_chars_in_str_fields(bad):
    # A free-form string carrying any control char is rejected (old kept),
    # so it can never reach config.toml.
    s = Settings(recording_dir="~/Meetings", recording_device="Mic")
    assert bp.apply_settings_patch(s, {"recording_dir": bad}).recording_dir == "~/Meetings"
    assert bp.apply_settings_patch(s, {"recording_device": bad}).recording_device == "Mic"


def test_is_safe_str_control_chars():
    assert bp._is_safe_str("~/Meetings") is True
    assert bp._is_safe_str("plain device name 2ch") is True
    assert bp._is_safe_str("has\ttab") is False
    assert bp._is_safe_str("has\x0cformfeed") is False
    assert bp._is_safe_str("has\x7fdel") is False
    assert bp._is_safe_str('has"quote') is False
    assert bp._is_safe_str("has\\backslash") is False


@pytest.mark.parametrize("surrogate", ["\ud800", "\udfff", "pre\udc00post"])
def test_is_safe_str_rejects_lone_surrogates(surrogate):
    # Lone UTF-16 surrogates are not utf-8 encodable and would truncate
    # config.toml to zero bytes on save — reject them at the patch boundary.
    assert bp._is_safe_str(surrogate) is False


def test_apply_patch_rejects_surrogate_in_str_fields():
    s = Settings(recording_dir="~/Meetings", recording_device="Mic")
    patched = bp.apply_settings_patch(s, {"recording_dir": "\ud800"})
    assert patched.recording_dir == "~/Meetings"
    patched2 = bp.apply_settings_patch(s, {"recording_device": "bad\udfffdev"})
    assert patched2.recording_device == "Mic"


# --- CLEANUP-PY: screenshot_displays dedup -------------------------------


def test_screenshot_displays_uses_config_source_of_truth():
    from whisper_daemon import config

    # No local duplicate set — the validator reuses config's constant.
    assert not hasattr(bp, "_VALID_SCREENSHOT_DISPLAYS")
    assert bp.VALID_SCREENSHOT_DISPLAYS is config.VALID_SCREENSHOT_DISPLAYS
    # And it still validates correctly through the single source.
    s = Settings()
    for v in config.VALID_SCREENSHOT_DISPLAYS:
        assert bp.apply_settings_patch(s, {"screenshot_displays": v}).screenshot_displays == v


# --- F5: send-once caption selection -------------------------------------


def _res(text):
    return {"text": text}


def test_select_unsent_results_returns_only_new_tail():
    all_results = [(0.0, _res("a")), (1.0, _res("b"))]
    out = bp.select_unsent_results(all_results, 0)
    assert out == [(0, 0.0, "a"), (1, 1.0, "b")]
    # After marking two sent, appending one yields only the new one at its
    # true index — no duplicates, no reordering.
    all_results.append((2.0, _res("c")))
    out2 = bp.select_unsent_results(all_results, 2)
    assert out2 == [(2, 2.0, "c")]


def test_select_unsent_results_skips_empty_text():
    all_results = [(0.0, _res("  ")), (1.0, _res("")), (2.0, _res("real"))]
    out = bp.select_unsent_results(all_results, 0)
    assert out == [(2, 2.0, "real")]


def test_select_unsent_results_nothing_new():
    all_results = [(0.0, _res("a"))]
    assert bp.select_unsent_results(all_results, 1) == []
    assert bp.select_unsent_results([], 0) == []


def test_apply_patch_never_mutates_via_replace():
    # sanity: apply uses dataclasses.replace semantics (fields equal except patched)
    s = Settings()
    result = bp.apply_settings_patch(s, {"diarize": True})
    changed = {
        f.name
        for f in dataclasses.fields(Settings)
        if getattr(s, f.name) != getattr(result, f.name)
    }
    assert changed == {"diarize"}


def test_settings_to_dict_includes_language():
    assert bp.settings_to_dict(Settings())["recording_language"] == "auto"


def test_apply_patch_accepts_valid_language():
    s = Settings()
    assert bp.apply_settings_patch(s, {"recording_language": "pl"}).recording_language == "pl"


def test_apply_patch_rejects_invalid_language():
    s = Settings(recording_language="pl")
    # unknown code is skipped — the old value is kept
    assert bp.apply_settings_patch(s, {"recording_language": "xx"}).recording_language == "pl"
    assert bp.apply_settings_patch(s, {"recording_language": 5}).recording_language == "pl"
