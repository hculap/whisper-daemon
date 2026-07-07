"""Tests for ffmpeg PATH resolution (the launchd minimal-PATH fix)."""

import os
import stat

import pytest

from whisper_daemon import ffmpeg_path
from whisper_daemon.ffmpeg_path import ensure_ffmpeg_on_path


def _make_fake_ffmpeg(directory) -> str:
    """Create an executable file named 'ffmpeg' in directory; return its path."""
    path = directory / "ffmpeg"
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return str(path)


def test_returns_existing_ffmpeg_without_touching_path(tmp_path, monkeypatch):
    """When ffmpeg is already on PATH, return it and leave PATH unchanged."""
    _make_fake_ffmpeg(tmp_path)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setattr(ffmpeg_path, "COMMON_BIN_DIRS", ())

    result = ensure_ffmpeg_on_path()

    assert result == str(tmp_path / "ffmpeg")
    assert os.environ["PATH"] == str(tmp_path)


def test_augments_path_when_ffmpeg_in_common_dir(tmp_path, monkeypatch):
    """When ffmpeg is only in a common install prefix, add it to PATH and find it."""
    brew_dir = tmp_path / "opt-homebrew-bin"
    brew_dir.mkdir()
    _make_fake_ffmpeg(brew_dir)

    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))  # ffmpeg not reachable here
    monkeypatch.setattr(ffmpeg_path, "COMMON_BIN_DIRS", (str(brew_dir),))

    result = ensure_ffmpeg_on_path()

    assert result == str(brew_dir / "ffmpeg")
    assert str(brew_dir) in os.environ["PATH"].split(os.pathsep)


def test_returns_none_when_ffmpeg_absent(tmp_path, monkeypatch):
    """No ffmpeg anywhere → None, and no crash."""
    empty = tmp_path / "empty"
    empty.mkdir()
    missing = tmp_path / "does-not-exist"
    monkeypatch.setenv("PATH", str(empty))
    monkeypatch.setattr(ffmpeg_path, "COMMON_BIN_DIRS", (str(missing),))

    assert ensure_ffmpeg_on_path() is None


def test_does_not_duplicate_existing_common_dir(tmp_path, monkeypatch):
    """A common dir already on PATH (but without ffmpeg) is not appended twice."""
    d = tmp_path / "bin"
    d.mkdir()
    monkeypatch.setenv("PATH", str(d))
    monkeypatch.setattr(ffmpeg_path, "COMMON_BIN_DIRS", (str(d),))

    ensure_ffmpeg_on_path()

    assert os.environ["PATH"].split(os.pathsep).count(str(d)) == 1
