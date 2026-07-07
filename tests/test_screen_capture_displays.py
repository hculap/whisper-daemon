"""Tests for screenshot_displays filtering in ScreenCapture.

These mock _get_display_ids and _capture_display so no Quartz calls happen.
"""

from pathlib import Path

from whisper_daemon import screen_capture as sc
from whisper_daemon.screen_capture import ScreenCapture


def _make_capture(tmp_path, displays):
    return ScreenCapture(Path(tmp_path), displays=displays)


def test_all_displays_captures_every_id(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "_get_display_ids", lambda: [1, 2, 3])
    cap = _make_capture(tmp_path, "all")

    called = []
    monkeypatch.setattr(
        cap, "_capture_display", lambda did, num, ts: called.append(did)
    )
    cap._capture_all_displays()
    assert called == [1, 2, 3]


def test_primary_captures_only_first(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "_get_display_ids", lambda: [1, 2, 3])
    cap = _make_capture(tmp_path, "primary")

    called = []
    monkeypatch.setattr(
        cap, "_capture_display", lambda did, num, ts: called.append(did)
    )
    cap._capture_all_displays()
    assert called == [1]


def test_default_is_all(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "_get_display_ids", lambda: [7, 8])
    cap = ScreenCapture(Path(tmp_path))  # no displays arg

    called = []
    monkeypatch.setattr(
        cap, "_capture_display", lambda did, num, ts: called.append(did)
    )
    cap._capture_all_displays()
    assert called == [7, 8]


def test_unknown_displays_value_treated_as_all(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "_get_display_ids", lambda: [1, 2, 3])
    cap = _make_capture(tmp_path, "bogus")

    called = []
    monkeypatch.setattr(
        cap, "_capture_display", lambda did, num, ts: called.append(did)
    )
    cap._capture_all_displays()
    assert called == [1, 2, 3]
