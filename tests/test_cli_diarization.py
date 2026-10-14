from types import SimpleNamespace
import sys
import numpy as np
import pytest
from whisper_daemon import __main__ as cli


def test_unavailable_speaker_model_is_visible_and_plain_recording_stays_enabled(monkeypatch, capsys):
    from whisper_daemon import model_policy
    monkeypatch.setattr(model_policy, 'diarization_status', lambda: {'available':False,'message':'Speaker model unavailable offline'})
    assert cli._speaker_model_ready(True) is False
    assert 'unavailable offline' in capsys.readouterr().out


def test_diarization_failure_preserves_completed_transcript(monkeypatch, capsys):
    def denied(*args, **kwargs): raise RuntimeError('401 Unauthorized')
    monkeypatch.setitem(sys.modules, 'whisper_daemon.diarizer', SimpleNamespace(diarize_file=denied, SpeakerSegment=object))
    transcript={'text':'Full transcript ending in42','segments':[{'start':0,'end':4,'text':'Full transcript ending in42'}]}
    assert cli._with_optional_speakers(transcript, 'test.wav', None) == transcript
    message=capsys.readouterr().out
    assert 'preserved' in message and '401 Unauthorized' in message


def test_live_embedding_failure_reaches_cli_fallback(monkeypatch):
    from whisper_daemon import diarizer
    def broken_model(*args, **kwargs): raise RuntimeError('embedding unavailable')
    monkeypatch.setattr(diarizer, '_get_embedding_model', lambda:broken_model)
    with pytest.raises(RuntimeError, match='embedding unavailable'):
        diarizer.SpeakerTracker().identify(np.ones(48000, dtype=np.float32),0)


def test_cli_realtime_embedding_failure_falls_back_to_batch(tmp_path, monkeypatch):
    import json
    from click.testing import CliRunner
    from whisper_daemon import diarizer, meeting_recorder, model_policy, recording_store, transcriber

    class StopAfterOneQueue(recording_store.DiskChunkQueue):
        got_chunk = False

        def get(self, timeout=0):
            if timeout == 0.5 and self.got_chunk:
                raise KeyboardInterrupt
            chunk = super().get(timeout)
            if timeout == 0.5:
                self.got_chunk = True
            return chunk

    class Recorder:
        error = None

        def __init__(self, jobs, raw_sink, **kwargs):
            self.jobs = jobs
            self.raw_sink = raw_sink

        def start(self):
            audio = np.full(32000, 0.1, dtype=np.float32)
            self.raw_sink(audio)
            self.jobs.put(meeting_recorder.AudioChunk(audio, 0, 2))

        def stop(self):
            pass

        def abort(self):
            pass

    def broken_embedding(waveform):
        raise RuntimeError("embedding runtime failure")

    batch_calls = []

    def batch(path, **kwargs):
        batch_calls.append(path)
        return [diarizer.SpeakerSegment(0, 0, 2)]

    transcript = {
        "text": "preserved CLI recording",
        "segments": [{"start": 0, "end": 2, "text": "preserved CLI recording"}],
        "language": "en",
    }
    monkeypatch.setattr(cli, "_setup_logging", lambda *args: None)
    monkeypatch.setattr(model_policy, "diarization_status", lambda: {"available": True, "message": "ready"})
    monkeypatch.setattr(transcriber, "preload_model", lambda *args: None)
    monkeypatch.setattr(transcriber, "transcribe_full", lambda *args, **kwargs: transcript)
    monkeypatch.setattr(diarizer, "_get_embedding_model", lambda: broken_embedding)
    monkeypatch.setattr(diarizer, "diarize_file", batch)
    monkeypatch.setattr(meeting_recorder, "MeetingRecorder", Recorder)
    monkeypatch.setattr(recording_store, "DiskChunkQueue", StopAfterOneQueue)

    output = tmp_path / "meeting.txt"
    result = CliRunner().invoke(cli.cli, ["record", str(output), "--diarize",
        "--diarize-mode", "realtime", "--format", "txt,json"])
    assert result.exit_code == 0, result.output
    assert len(batch_calls) == 1
    assert "Using final batch labels" in result.output
    saved = json.loads(output.with_suffix(".json").read_text())
    assert saved["text"] == transcript["text"]
    assert saved["speakers"] == [{"id": 0, "label": "Speaker 1"}]
    assert saved["segments"][0]["speaker"] == 0
    assert transcript["text"] in output.read_text()
