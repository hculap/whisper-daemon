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


def browser_recorder(store, monkeypatch):
    from whisper_daemon import meeting_recorder as mod
    class VAD:
        def reset_states(self): pass
        def __call__(self, samples): return 1
    monkeypatch.setattr(mod, 'SileroVAD', VAD)
    jobs = queue.Queue()
    recorder = mod.MeetingRecorder(jobs, source_label='browser',
        raw_sink=lambda audio, position: store.append('browser', audio, at=position),
        sample_filter=store.unpaused_audio)
    recorder.start_without_device()
    return recorder, jobs


def test_reconnect_replay_uses_capture_positions_not_arrival_time(tmp_path, monkeypatch):
    now = [0.0]
    store = RecordingStore(tmp_path, clock=lambda: now[0])
    recorder, jobs = browser_recorder(store, monkeypatch)
    for position in range(0, 16384, 512):
        now[0] = (position + 512) / 16000
        recorder.feed_audio(np.full(512, .1, dtype=np.float32), position=position)
    now[0] = 6.144
    for position in range(16384, 98304, 512):
        recorder.feed_audio(np.full(512, .2, dtype=np.float32), position=position)
    recorder.stop_without_device()
    store.finish(duration=6.144)
    assert store.frames['browser'] == 98304
    assert store.meta['gaps'] == []
    assert np.all(np.fromfile(tmp_path / 'browser.f32', dtype='<f4')[16384:] == .2)


def test_capture_time_pause_excludes_replayed_pcm_and_keeps_earlier_audio(tmp_path, monkeypatch):
    now = [0.0]
    store = RecordingStore(tmp_path, clock=lambda: now[0])
    recorder, jobs = browser_recorder(store, monkeypatch)
    now[0] = 2; store.pause(True)
    # Late, but captured before pause: it must survive.
    recorder.feed_audio(np.full(32000, .1, dtype=np.float32), position=0)
    now[0] = 4; store.pause(False)
    now[0] = 6
    recorder.feed_audio(np.full(32000, .9, dtype=np.float32), position=32000)
    recorder.feed_audio(np.full(32000, .2, dtype=np.float32), position=64000)
    recorder.stop_without_device()
    store.finish(duration=6)
    raw = np.fromfile(tmp_path / 'browser.f32', dtype='<f4')
    assert np.all(raw[:32000] == .1)
    assert not np.any(raw[32000:64000])
    assert np.all(raw[64000:] == .2)
    while not jobs.empty():
        chunk = jobs.get_nowait()
        if chunk is not None: assert not np.any(chunk.audio == .9)
    assert store.meta['gaps'] == [{'source':'browser', 'start':2., 'end':4., 'reason':'paused'}]


def test_one_sample_tail_finishes_and_trailing_loss_is_reported(tmp_path, monkeypatch):
    store = RecordingStore(tmp_path, clock=lambda: 0)
    recorder, _ = browser_recorder(store, monkeypatch)
    recorder.feed_audio(np.zeros(320001, dtype=np.float32), position=0)
    recorder.stop_without_device()
    store.finish(duration=21)
    assert store.frames['browser'] == 320001
    assert store.meta['gaps'][-1] == {'source':'browser', 'start':320001/16000,
                                    'end':21., 'reason':'missing audio'}


def test_delivery_latency_at_the_end_is_not_reported_as_missing_audio(tmp_path):
    store = RecordingStore(tmp_path, clock=lambda: 0, sources=('mic', 'browser'))
    store.append('mic', np.zeros(160000, dtype=np.float32), at=0)
    store.append('browser', np.zeros(159000, dtype=np.float32), at=0)
    store.finish(duration=10.06)  # wall-clock stop lands after the last packets
    assert store.meta['gaps'] == []
    assert store.meta['duration'] == 10.06


def test_callback_liveness_detects_dead_mic_even_when_paused(monkeypatch, tmp_path):
    store = RecordingStore(tmp_path, clock=lambda: 0)
    recorder, _ = browser_recorder(store, monkeypatch)
    recorder._source_label = 'mic'
    recorder._paused = lambda: True
    recorder._last_audio_at = 100
    assert recorder.check_liveness(111)
    assert recorder.needs_recovery
    assert not recorder.check_liveness(112)  # one alert per failure
