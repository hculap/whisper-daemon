from types import SimpleNamespace

import numpy as np
import pytest

from whisper_daemon import diarizer


def test_release_models_drops_references_and_flushes_mps(monkeypatch):
    calls = []
    monkeypatch.setattr(diarizer, "_pipeline", object())
    monkeypatch.setattr(diarizer, "_embedding_model", object())
    monkeypatch.setattr(diarizer.gc, "collect", lambda: calls.append("gc"))
    monkeypatch.setattr(diarizer.torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(diarizer.torch.mps, "synchronize", lambda: calls.append("sync"))
    monkeypatch.setattr(diarizer.torch.mps, "empty_cache", lambda: calls.append("empty"))

    diarizer.release_models()

    assert diarizer._pipeline is None
    assert diarizer._embedding_model is None
    assert calls == ["gc", "sync", "empty"]


def _patch_file_decode(monkeypatch):
    from whisper_daemon import ffmpeg_path
    import subprocess

    monkeypatch.setattr(ffmpeg_path, "ensure_ffmpeg_on_path", lambda: "/opt/ffmpeg")

    def decode(*args, stdout, **kwargs):
        stdout.write(np.ones(16000, dtype="<f4").tobytes())
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", decode)
    monkeypatch.setattr(
        diarizer.np,
        "memmap",
        lambda *args, **kwargs: np.ones(16000, dtype=np.float32),
    )


def test_diarize_file_releases_models_after_success(monkeypatch):
    _patch_file_decode(monkeypatch)
    segment = diarizer.SpeakerSegment(0, 0, 1)
    calls = []
    monkeypatch.setattr(diarizer, "diarize_batch", lambda *args, **kwargs: [segment])
    monkeypatch.setattr(diarizer, "release_models", lambda: calls.append("release"))

    assert diarizer.diarize_file("example.wav") == [segment]
    assert calls == ["release"]


def test_diarize_file_releases_models_when_inference_fails(monkeypatch):
    _patch_file_decode(monkeypatch)
    calls = []

    def fail(*args, **kwargs):
        raise RuntimeError("diarization failed")

    monkeypatch.setattr(diarizer, "diarize_batch", fail)
    monkeypatch.setattr(diarizer, "release_models", lambda: calls.append("release"))

    with pytest.raises(RuntimeError, match="diarization failed"):
        diarizer.diarize_file("example.wav")
    assert calls == ["release"]


def test_diarize_file_can_keep_models_for_a_batch(monkeypatch):
    _patch_file_decode(monkeypatch)
    calls = []
    monkeypatch.setattr(diarizer, "diarize_batch", lambda *args, **kwargs: [])
    monkeypatch.setattr(diarizer, "release_models", lambda: calls.append("release"))

    diarizer.diarize_file("example.wav", release=False)
    assert calls == []
