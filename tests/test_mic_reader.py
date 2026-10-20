"""The microphone is read in blocking mode with explicit session positions."""
import contextlib
import queue
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import sounddevice as sd

from whisper_daemon import meeting_recorder as mod
from whisper_daemon.recording_store import RecordingStore

RATE = 16000
BLOCK = mod.BLOCK_SIZE


class FakeDevice:
    """Owns the clock: produced samples advance it; lost audio advances it alone."""

    def __init__(self):
        self.now = 0.0
        self.streams = []

    def clock(self):
        return self.now

    def produce(self, samples, value=0.1, seconds=None):
        self.streams[-1].push(np.full((samples, 1), value, dtype=np.float32))
        self.now += samples / RATE if seconds is None else seconds

    def lose(self, samples, overflow=True):
        self.now += samples / RATE
        self.streams[-1].overflow = overflow


class FakeStream:
    def __init__(self, owner, **kwargs):
        self.kwargs = kwargs
        self.buffer = np.zeros((0, 1), dtype=np.float32)
        self.overflow = False
        self.closed = False
        self.error = None
        self.lock = threading.Lock()
        owner.streams.append(self)

    def push(self, data):
        with self.lock:
            self.buffer = np.concatenate([self.buffer, data])

    def start(self): pass
    def stop(self): pass
    def close(self): self.closed = True

    @property
    def read_available(self):
        with self.lock:
            return len(self.buffer)

    def read(self, frames):
        if self.error:
            raise self.error
        with self.lock:
            data, self.buffer = self.buffer[:frames], self.buffer[frames:]
            overflowed, self.overflow = self.overflow, False
        return data, overflowed


@pytest.fixture
def device(monkeypatch):
    device = FakeDevice()

    class VAD:
        def reset_states(self): pass
        def __call__(self, samples): return 0

    monkeypatch.setattr(mod, "SileroVAD", VAD)
    monkeypatch.setattr(mod.sd, "InputStream", lambda **kwargs: FakeStream(device, **kwargs))
    monkeypatch.setattr(mod, "audio_system", SimpleNamespace(
        stream_open=contextlib.nullcontext, ensure_fresh=lambda force=False: True,
        register_stream=lambda stream: None, unregister_stream=lambda stream: None,
        mark_dirty=lambda reason: None))
    return device


def make(tmp_path, device, monkeypatch, threaded=False):
    if not threaded:
        monkeypatch.setattr(mod.MeetingRecorder, "_start_reader", lambda self: None)
    store = RecordingStore(tmp_path, clock=device.clock, started_at=0.0, sources=("mic",))
    recorder = mod.MeetingRecorder(
        queue.Queue(), raw_sink=lambda audio, position: store.append("mic", audio, at=position),
        sample_filter=store.unpaused_audio, started_at=0.0, clock=device.clock)
    recorder.start()
    return recorder, store


def finish(recorder, store, device):
    recorder.stop()
    store.finish(duration=device.now)
    return store.meta["gaps"]


def test_stream_is_opened_for_blocking_reads_with_a_deep_buffer(tmp_path, device, monkeypatch):
    make(tmp_path, device, monkeypatch)
    kwargs = device.streams[-1].kwargs
    assert "callback" not in kwargs
    assert kwargs["latency"] == mod.READ_LATENCY
    # Real meetings stalled the reader for ~5 s; the device must buffer that.
    assert mod.READ_LATENCY >= 4.0


def test_a_stalled_reader_loses_no_audio(tmp_path, device, monkeypatch):
    recorder, store = make(tmp_path, device, monkeypatch)
    device.produce(32 * BLOCK)
    recorder.drain(device.streams[-1])
    device.produce(64 * BLOCK)  # reader busy for ~2 s: the device keeps buffering
    recorder.drain(device.streams[-1])
    assert finish(recorder, store, device) == []
    assert store.frames["mic"] == 96 * BLOCK


def test_lost_device_audio_is_a_gap_where_it_happened(tmp_path, device, monkeypatch):
    recorder, store = make(tmp_path, device, monkeypatch)
    device.produce(32 * BLOCK)
    recorder.drain(device.streams[-1])
    device.lose(16 * BLOCK)
    device.produce(32 * BLOCK)
    recorder.drain(device.streams[-1])
    gaps = finish(recorder, store, device)
    assert gaps == [{"source": "mic", "start": 32 * BLOCK / RATE, "end": 48 * BLOCK / RATE,
                     "reason": "missing audio"}]
    assert recorder._overflows == 1


def test_falling_behind_without_an_overflow_flag_still_marks_the_loss(tmp_path, device, monkeypatch):
    recorder, store = make(tmp_path, device, monkeypatch)
    device.produce(32 * BLOCK)
    recorder.drain(device.streams[-1])
    device.lose(16 * BLOCK, overflow=False)
    device.produce(32 * BLOCK)
    recorder.drain(device.streams[-1])
    assert [g["start"] for g in finish(recorder, store, device)] == [32 * BLOCK / RATE]


def test_jitter_under_the_tolerance_keeps_the_track_contiguous(tmp_path, device, monkeypatch):
    recorder, store = make(tmp_path, device, monkeypatch)
    device.produce(32 * BLOCK)
    recorder.drain(device.streams[-1])
    device.lose(4 * BLOCK, overflow=False)  # 128 ms late, below REANCHOR_SAMPLES
    device.produce(32 * BLOCK)
    recorder.drain(device.streams[-1])
    recorder.stop()
    assert store.frames["mic"] == 64 * BLOCK
    assert store.meta["gaps"] == []


def test_a_reopened_stream_never_moves_the_track_backwards(tmp_path, device, monkeypatch):
    recorder, store = make(tmp_path, device, monkeypatch)
    # The old device ran fast: its positions are ahead of the wall clock.
    device.produce(32 * BLOCK, seconds=0.9)
    recorder.drain(device.streams[-1])
    assert recorder.recover_device()
    device.produce(32 * BLOCK)
    recorder.drain(device.streams[-1])
    assert recorder.error is None
    assert store.frames["mic"] == 64 * BLOCK


def test_pause_keeps_mic_audio_out_and_marks_the_paused_time(tmp_path, device, monkeypatch):
    recorder, store = make(tmp_path, device, monkeypatch)
    device.produce(32 * BLOCK)
    recorder.drain(device.streams[-1])
    store.pause(True)
    device.produce(32 * BLOCK, value=0.5)
    recorder.drain(device.streams[-1])
    store.pause(False)
    device.produce(32 * BLOCK)
    recorder.drain(device.streams[-1])
    gaps = finish(recorder, store, device)
    assert gaps == [{"source": "mic", "start": 32 * BLOCK / RATE, "end": 64 * BLOCK / RATE, "reason": "paused"}]
    assert not np.any(np.fromfile(tmp_path / "mic.f32", dtype="<f4") == 0.5)


def test_an_all_zero_stream_requests_recovery(tmp_path, device, monkeypatch):
    monkeypatch.setattr(mod, "DEAD_STREAM_BLOCKS", 3)
    recorder, _ = make(tmp_path, device, monkeypatch)
    device.produce(3 * BLOCK, value=0.0)
    recorder.drain(device.streams[-1])
    assert recorder.needs_recovery


def test_a_failing_read_arms_recovery_and_ends_the_reader(tmp_path, device, monkeypatch):
    recorder, _ = make(tmp_path, device, monkeypatch)
    stream = device.streams[-1]
    stream.error = sd.PortAudioError("device lost")
    device.produce(BLOCK)
    recorder._read_loop(stream, threading.Event())  # returns instead of spinning
    assert recorder.needs_recovery


def test_the_reader_thread_drains_the_device_and_stops_with_it(tmp_path, device, monkeypatch):
    recorder, store = make(tmp_path, device, monkeypatch, threaded=True)
    stream = device.streams[-1]
    device.produce(32 * BLOCK)
    deadline = time.monotonic() + 2
    while store.frames.get("mic", 0) < 32 * BLOCK and time.monotonic() < deadline:
        time.sleep(0.01)
    assert store.frames["mic"] == 32 * BLOCK
    reader = recorder._reader
    recorder.stop()
    assert not reader.is_alive()
    assert recorder._reader is None and stream.closed
    recorder.abort()  # idempotent after stop
