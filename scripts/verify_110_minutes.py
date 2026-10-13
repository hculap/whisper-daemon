"""Accelerated 110-minute two-source capture through the real recorder/store.

Runs without microphone/network/inference; validates sample integrity, silence,
short tail, crash-compatible exports and bounded RSS. Use a temporary directory.
"""
import json
import resource
import tempfile
import time
import wave
from pathlib import Path

import numpy as np
from whisper_daemon import meeting_recorder as mod
from whisper_daemon.recording_store import RecordingStore, DiskChunkQueue

class SilentVAD:
    def reset_states(self): pass
    def __call__(self, samples): return 0
mod.SileroVAD = SilentVAD

start = time.monotonic()
with tempfile.TemporaryDirectory(prefix='whisper-110min-') as directory:
    path = Path(directory)
    store = RecordingStore(path, clock=lambda: 0)
    jobs = DiskChunkQueue(path)
    recorders = []
    for source in ('mic', 'browser'):
        rec = mod.MeetingRecorder(jobs, source_label=source,
            raw_sink=lambda audio, s=source: store.append(s, audio, at=store.frames.get(s, 0)))
        rec.start_without_device()
        recorders.append(rec)
    baseline_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    total = 110 * 60 * 16000
    block = np.zeros(16000, dtype=np.float32)
    for second in range(110 * 60):
        block[:] = .001 if second % 2 else 0  # keep actual alternating silent audio
        for rec in recorders: rec.feed_audio(block)
    for rec in recorders: rec.stop_without_device()
    store.finish(duration=110 * 60)
    for source in ('mic', 'browser', 'recording'):
        with wave.open(str(path / f'{source}.wav')) as wav:
            assert wav.getnframes() == total
    for source in ('mic', 'browser'):
        raw = np.memmap(path / f'{source}.f32', dtype='<f4', mode='r')
        assert len(raw) == total
        for second in (0, 1, 3599, 6598, 6599):
            assert np.all(raw[second * 16000:(second+1) * 16000] == (.001 if second % 2 else 0))
        del raw
    rss_delta = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - baseline_rss
    assert rss_delta < 180 * 1024 * 1024, rss_delta
    report = {'test': 'accelerated_110_min_two_sources', 'seconds_per_source': 6600,
              'samples_per_source': total, 'rss_growth_bytes': rss_delta,
              'runtime_seconds': round(time.monotonic() - start, 2)}
    print(json.dumps(report))
