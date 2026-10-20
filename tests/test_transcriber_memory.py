from types import SimpleNamespace
from pathlib import Path
import tomllib

import numpy as np
import pytest

from whisper_daemon import transcriber


class _Watchdog:
    def __init__(self, events):
        self._events = events

    def cancel(self):
        self._events.append("cancel")


def test_project_requires_an_mlx_version_with_memory_cache_api():
    dependencies = tomllib.loads(Path("pyproject.toml").read_text())["project"]["dependencies"]
    assert "mlx>=0.31" in dependencies


def _install_fakes(monkeypatch, inference):
    events = []
    fake_mx = SimpleNamespace(
        set_cache_limit=lambda limit: events.append(("limit", limit)) or 99,
        get_active_memory=lambda: 123,
        get_cache_memory=lambda: 456,
        clear_cache=lambda: events.append("clear"),
    )
    monkeypatch.setattr(transcriber, "mx", fake_mx, raising=False)
    monkeypatch.setattr(transcriber, "_mlx_memory_configured", False, raising=False)
    monkeypatch.setattr(
        transcriber,
        "_hang_watchdog",
        lambda *args, **kwargs: _Watchdog(events),
    )

    def call(*args, **kwargs):
        events.append("infer")
        return inference(*args, **kwargs)

    monkeypatch.setattr(transcriber.mlx_whisper, "transcribe", call)
    return events


def test_mlx_cache_is_bounded_before_preload_and_cleared_after(monkeypatch):
    events = _install_fakes(
        monkeypatch,
        lambda *args, **kwargs: {"text": "", "segments": [], "language": "en"},
    )
    monkeypatch.setattr(transcriber, "_model_preloaded", False)

    transcriber.preload_model("test-model")

    assert events == [
        ("limit", transcriber.MLX_CACHE_LIMIT_BYTES),
        "infer",
        "cancel",
        "clear",
    ]
    assert transcriber._model_preloaded


@pytest.mark.parametrize("entrypoint", ["transcribe", "transcribe_full"])
def test_array_inference_clears_cache_when_inference_fails(monkeypatch, entrypoint):
    class PrimaryError(RuntimeError):
        pass

    def fail(*args, **kwargs):
        raise PrimaryError("decoder failed")

    events = _install_fakes(monkeypatch, fail)
    with pytest.raises(PrimaryError, match="decoder failed"):
        getattr(transcriber, entrypoint)(np.ones(16000, dtype=np.float32), model="test")

    assert events[-2:] == ["cancel", "clear"]


def test_file_inference_clears_cache_after_success(monkeypatch):
    events = _install_fakes(
        monkeypatch,
        lambda *args, **kwargs: {
            "text": " hello ",
            "segments": [{"text": " hello ", "start": 0, "end": 1}],
            "language": "en",
        },
    )
    monkeypatch.setattr(transcriber, "ensure_ffmpeg_on_path", lambda: "/opt/ffmpeg")
    monkeypatch.setattr(
        transcriber.mlx_whisper.audio,
        "load_audio",
        lambda path: np.ones(16000, dtype=np.float32),
    )

    result = transcriber.transcribe_file("example.wav", model="test")

    assert result["text"] == "hello"
    assert events[-2:] == ["cancel", "clear"]


def test_cleanup_failure_does_not_destroy_a_valid_transcript(monkeypatch):
    events = _install_fakes(
        monkeypatch,
        lambda *args, **kwargs: {"text": "kept", "segments": [], "language": "en"},
    )

    def broken_clear():
        events.append("clear")
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(transcriber.mx, "clear_cache", broken_clear)
    assert transcriber.transcribe(np.ones(16000, dtype=np.float32), model="test") == "kept"
    assert events[-2:] == ["cancel", "clear"]


def test_memory_stats_failure_does_not_skip_cache_cleanup(monkeypatch):
    events = _install_fakes(
        monkeypatch,
        lambda *args, **kwargs: {"text": "kept", "segments": [], "language": "en"},
    )
    monkeypatch.setattr(
        transcriber.mx,
        "get_active_memory",
        lambda: (_ for _ in ()).throw(RuntimeError("stats unavailable")),
    )

    assert transcriber.transcribe(np.ones(16000, dtype=np.float32), model="test") == "kept"
    assert "clear" in events
