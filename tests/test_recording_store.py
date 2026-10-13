import json
import os
import queue
import subprocess
import sys
import wave

import numpy as np
import pytest

from whisper_daemon.recording_store import RecordingStore, DiskChunkQueue
from whisper_daemon.meeting_recorder import AudioChunk


def test_continuous_sources_include_silence_and_share_timeline(tmp_path):
    store = RecordingStore(tmp_path, clock=lambda: 0)
    store.append('mic', np.ones(16000, dtype=np.float32), at=0)
    store.append('mic', np.zeros(16000, dtype=np.float32), at=16000)
    store.append('browser', np.full(8000, .5, dtype=np.float32), at=24000)
    store.finish(duration=2)
    for source in ('mic', 'browser'):
        with wave.open(str(tmp_path / f'{source}.wav')) as wav:
            assert wav.getnframes() == 32000
    browser = np.fromfile(tmp_path / 'browser.f32', dtype='<f4')
    assert np.all(browser[:24000] == 0)
    assert np.all(browser[24000:] == .5)
    assert json.loads((tmp_path / 'session.json').read_text())['gaps']


def test_invalid_pcm_never_reaches_disk(tmp_path):
    store = RecordingStore(tmp_path, clock=lambda: 0)
    with pytest.raises(ValueError):
        store.append('mic', np.array([np.nan], dtype=np.float32))
    assert not (tmp_path / 'mic.f32').exists()


def test_killed_writer_recovers_entire_raw_audio(tmp_path):
    code = '''import os, numpy as np
from whisper_daemon.recording_store import RecordingStore
s=RecordingStore(os.environ['TEST_RECORD_DIR'], clock=lambda:0)
s.append('mic', np.ones(16000*3,dtype=np.float32), at=0)
os._exit(9)
'''
    result = subprocess.run([sys.executable, '-c', code], env={**os.environ, 'TEST_RECORD_DIR': str(tmp_path)})
    assert result.returncode == 9
    recovered = RecordingStore.recover(tmp_path)
    assert recovered['duration'] == 3
    with wave.open(str(tmp_path / 'mic.wav')) as wav:
        assert wav.getnframes() == 48000


def test_disk_backlog_survives_restart_and_ack_is_durable(tmp_path):
    q = DiskChunkQueue(tmp_path)
    chunk = AudioChunk(np.ones(32000, dtype=np.float32), 5, 2, 'browser')
    q.put(chunk)
    q = DiskChunkQueue(tmp_path)
    got = q.get_nowait()
    assert got.source == 'browser' and got.start_time == 5
    assert np.array_equal(got.audio, chunk.audio)
    q.ack(got, {'text': 'hello', 'segments': []})
    q = DiskChunkQueue(tmp_path)
    with pytest.raises(queue.Empty): q.get_nowait()
    assert q.results()[0][1]['source'] == 'browser'


def test_recorder_persists_silence_and_short_final_packet_before_vad(tmp_path, monkeypatch):
    from whisper_daemon import meeting_recorder as mod
    class VAD:
        def reset_states(self): pass
        def __call__(self, samples): return 0
    monkeypatch.setattr(mod, 'SileroVAD', VAD)
    store = RecordingStore(tmp_path, clock=lambda: 0)
    jobs = DiskChunkQueue(tmp_path)
    recorder = mod.MeetingRecorder(jobs, raw_sink=lambda audio: store.append('browser', audio, at=store.frames.get('browser', 0)))
    recorder.start_without_device()
    recorder.feed_audio(np.zeros(16513, dtype=np.float32))
    recorder.stop_without_device()
    store.finish()
    assert (tmp_path / 'browser.f32').stat().st_size == 16513 * 4
    with pytest.raises(queue.Empty): jobs.get_nowait()


def test_paused_audio_is_omitted_and_explicit_gap_keeps_wall_time(tmp_path):
    now = [0.0]
    store = RecordingStore(tmp_path, clock=lambda: now[0])
    store.append('mic', np.ones(16000, dtype=np.float32), at=0)
    now[0] = 1
    store.pause(True)
    now[0] = 4
    store.pause(False)
    store.append('mic', np.ones(16000, dtype=np.float32))
    store.finish()
    assert store.frames['mic'] == 64000
    assert store.meta['gaps'][0]['start'] == 1
    assert store.meta['gaps'][0]['end'] == 3
