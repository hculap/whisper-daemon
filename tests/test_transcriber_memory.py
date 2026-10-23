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



LOOP = {"text": " że to jest to," * 43, "language": "pl", "segments": [
    {"text": " Dziś," + " że to jest to," * 43, "start": 0, "end": 14.7, "compression_ratio": 20.85}]}
SPEECH = {"text": " Dobra. Wszystkie prawa zastrzeżone.", "language": "pl", "segments": [
    {"text": " Dobra.", "start": 0, "end": 1, "compression_ratio": 0.6},
    {"text": " Wszystkie prawa zastrzeżone.", "start": 1, "end": 2, "compression_ratio": 0.9}]}


def _decode(monkeypatch, *results):
    calls, pending = [], list(results)
    _install_fakes(monkeypatch, lambda audio, **kwargs: calls.append(kwargs) or pending.pop(0))
    monkeypatch.setattr(transcriber, "ensure_ffmpeg_on_path", lambda: "/opt/ffmpeg")
    monkeypatch.setattr(transcriber.mlx_whisper.audio, "load_audio", lambda path: np.ones(16000, dtype=np.float32))
    return calls


def test_a_clean_chunk_is_decoded_once_greedily_without_hallucinations(monkeypatch):
    calls = _decode(monkeypatch, SPEECH)
    result = transcriber.transcribe_full(np.ones(16000, dtype=np.float32), model="test")
    assert [c["temperature"] for c in calls] == [0]
    assert result["text"] == "Dobra." and [s["text"] for s in result["segments"]] == ["Dobra."]


def test_a_looping_chunk_is_retried_at_low_temperatures(monkeypatch):
    calls = _decode(monkeypatch, LOOP, SPEECH)
    result = transcriber.transcribe_full(np.ones(16000, dtype=np.float32), model="test")
    assert [c["temperature"] for c in calls] == [0, transcriber.RETRY_TEMPERATURES]
    assert max(transcriber.RETRY_TEMPERATURES) <= 0.4  # higher ones invent words
    assert calls[1]["compression_ratio_threshold"] == 2.4 and calls[1]["logprob_threshold"] is None
    assert result["text"] == "Dobra."


def test_a_looping_retry_keeps_the_greedy_text_collapsed(monkeypatch):
    calls = _decode(monkeypatch, LOOP, {**LOOP, "segments": [{**LOOP["segments"][0], "text": " laborator" * 60}]})
    result = transcriber.transcribe_full(np.ones(16000, dtype=np.float32), model="test")
    assert len(calls) == 2
    assert result["text"] == "Dziś, że to jest to,"


@pytest.mark.parametrize("entrypoint", ["transcribe", "transcribe_file"])
def test_dictation_and_files_stay_greedy(monkeypatch, entrypoint):
    calls = _decode(monkeypatch, LOOP)
    audio = "example.wav" if entrypoint == "transcribe_file" else np.ones(16000, dtype=np.float32)
    result = getattr(transcriber, entrypoint)(audio, model="test")
    assert [c["temperature"] for c in calls] == [0]
    if entrypoint == "transcribe_file":
        assert result["text"] == "Dziś, że to jest to,"
