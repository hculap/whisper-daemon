"""Continuous, recoverable source audio and a disk-backed transcription backlog.

Raw little-endian float32 files are authoritative. They have no header to corrupt
on process termination. Exports are streamed in bounded blocks, never concatenated
in RAM. A process crash loses no samples already written; fsync limits the power
failure window to one second. Raw sources are retained for recovery/reprocessing.
"""
from collections.abc import Iterator
import json
import os
from pathlib import Path
import queue
import threading
import time
import uuid
import wave

import numpy as np

RATE = 16000
SOURCES = ('mic', 'browser')
# A source ending this close to the wall-clock stop is delivery latency (last
# packet/device block in flight), not missing audio. Matches append()'s slack.
TRAILING_SLACK = RATE // 4


def atomic_json(path: Path, value: object) -> None:
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False, encoding='utf-8') as file:
        temp = Path(file.name)
        try:
            json.dump(value, file, ensure_ascii=False, allow_nan=False)
            file.flush()
            os.fsync(file.fileno())
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
    temp.replace(path)


class RecordingStore:
    def __init__(self, path: Path, *, clock=time.monotonic, started_at=None, sources=()):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.clock = clock
        self.started_at = clock() if started_at is None else started_at
        self.lock = threading.RLock()
        self.files = {}
        self.frames = {}
        self.enabled_sources = set(sources)
        self.levels = {}
        self.last_audio = {}
        self.last_sync = clock()
        self.meta = {'version': 1, 'state': 'recording', 'sample_rate': RATE,
                     'sources': {}, 'gaps': [], 'pauses': [], 'errors': []}
        self._save()

    def _save(self):
        self.meta['sources'] = dict(self.frames)
        atomic_json(self.path / 'session.json', self.meta)

    def append(self, source, samples, *, at=None):
        """Write once before VAD. `at` is an absolute session sample position.

        An explicit position detects duplicates and packet gaps. Device input
        uses its sample clock, with wall time only to align startup/recovery.
        """
        samples = np.asarray(samples, dtype='<f4').reshape(-1)
        if source not in SOURCES or not np.isfinite(samples).all() or np.any(np.abs(samples) > 1.01):
            raise ValueError('Invalid source or PCM samples')
        with self.lock:
            cursor = self.frames.get(source, 0)
            if at is None:
                expected = max(0, round((self.clock() - self.started_at) * RATE) - len(samples))
                at = expected if expected > cursor + RATE // 4 else cursor
            if not isinstance(at, int) or at < 0:
                raise ValueError('Invalid sample position')
            if at < cursor:
                raise ValueError('Duplicate or overlapping PCM')
            if source not in self.files:
                self.files[source] = (self.path / f'{source}.f32').open('xb', buffering=0)
            file = self.files[source]
            if at > cursor:
                self._record_gap(source, cursor, at)
            file.seek(at * 4)
            payload = samples.tobytes()
            if file.write(payload) != len(payload):
                raise OSError("Short write while storing source audio")
            self.frames[source] = at + len(samples)
            self.levels[source] = float(np.sqrt(np.mean(samples ** 2))) if len(samples) else 0
            self.last_audio[source] = self.clock()
            if self.clock() - self.last_sync >= 1:
                self.flush()
            return at / RATE

    def health(self):
        with self.lock:
            return {s: {'rms': self.levels.get(s, 0),
                        'age': max(0, self.clock() - self.last_audio.get(s, self.started_at)),
                        'seconds': self.frames.get(s, 0) / RATE} for s in SOURCES}

    def flush(self):
        with self.lock:
            for file in self.files.values():
                os.fsync(file.fileno())
            self.last_sync = self.clock()
            self._save()

    def pause(self, paused, *, at=None):
        with self.lock:
            self.meta['pauses'].append({'time': self.clock() - self.started_at if at is None else at, 'paused': paused})
            self.flush()

    def _intervals(self, start: int, end: int) -> Iterator[tuple[int, int, bool]]:
        """Split a capture-time interval into paused/unpaused sample ranges."""
        paused = False
        cursor = start
        for event in self.meta['pauses']:
            position = round(event['time'] * RATE)
            if position <= start:
                paused = event['paused']
                continue
            if position >= end: break
            yield cursor, position, paused
            cursor, paused = position, event['paused']
        if cursor < end: yield cursor, end, paused

    def unpaused_audio(self, samples: np.ndarray, position: int) -> list[tuple[int, np.ndarray]]:
        # Take the interval snapshot under the same lock used by Pause. No
        # receipt-time flag: replay may arrive on either side of a later pause.
        with self.lock:
            intervals = list(self._intervals(position, position + len(samples)))
        return [(start, samples[start-position:end-position])
                for start, end, paused in intervals if not paused]

    def _record_gap(self, source, start, end):
        for first, last, paused in self._intervals(start, end):
            self.meta['gaps'].append({'source': source, 'start': first / RATE,
                'end': last / RATE, 'reason': 'paused' if paused else 'missing audio'})

    def finish(self, duration=None, state='complete'):
        with self.lock:
            self.flush()
            for file in self.files.values():
                file.close()
            self.files.clear()
            frames = max(max(self.frames.values(), default=0), round((duration or 0) * RATE))
            for source in self.enabled_sources | self.frames.keys():
                if frames - self.frames.get(source, 0) > TRAILING_SLACK:
                    self._record_gap(source, self.frames.get(source, 0), frames)
            self.meta.update(state=state, duration=frames / RATE)
            self._save()
            self.export(self.path, frames)
            return self.meta

    @staticmethod
    def export(path, frames):
        path = Path(path)
        sources = [s for s in SOURCES if (path / f'{s}.f32').exists()]
        readers = {s: (path / f'{s}.f32').open('rb') for s in sources}
        writers = {}
        try:
            for name in [*sources, 'recording']:
                writer = wave.open(str(path / f'{name}.wav.tmp'), 'wb')
                writer.setparams((1, 2, RATE, 0, 'NONE', 'not compressed'))
                writers[name] = writer
            for start in range(0, frames, RATE):
                size = min(RATE, frames - start)
                mixed = np.zeros(size, dtype=np.float32)
                for source, reader in readers.items():
                    payload = reader.read(size * 4)
                    data = np.frombuffer(payload[:len(payload) // 4 * 4], dtype='<f4')
                    data = np.pad(data, (0, size - len(data)))
                    mixed += data
                    writers[source].writeframesraw((data.clip(-1, 1) * 32767).astype('<i2').tobytes())
                # Equal gain keeps simultaneous sources from clipping.
                mixed /= max(1, len(sources))
                writers['recording'].writeframesraw((mixed.clip(-1, 1) * 32767).astype('<i2').tobytes())
        finally:
            for reader in readers.values(): reader.close()
            for writer in writers.values(): writer.close()
        for name in writers:
            (path / f'{name}.wav.tmp').replace(path / f'{name}.wav')

    @classmethod
    def recover(cls, path):
        path = Path(path)
        metadata_path = path / 'session.json'
        meta = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        frames = {s: (path / f'{s}.f32').stat().st_size // 4 for s in SOURCES
                  if (path / f'{s}.f32').exists()}
        if not frames:
            raise ValueError('No recoverable source audio')
        duration_frames = max(max(frames.values()), round(meta.get('duration', 0) * RATE))
        cls.export(path, duration_frames)
        meta.update(state='recovered', sources=frames, duration=duration_frames / RATE)
        atomic_json(metadata_path, meta)
        return meta


class DiskChunkQueue:
    """Queue-compatible spool; pending audio stays on disk until its result is saved."""
    def __init__(self, path):
        self.path = Path(path) / 'chunks'
        self.path.mkdir(parents=True, exist_ok=True)
        self.condition = threading.Condition()
        self.inflight = set()

    def put(self, chunk):
        if chunk is None: return  # session lifecycle explicitly owns stopping
        name = f'{time.time_ns():020d}-{uuid.uuid4().hex}'
        with self.condition:
            with (self.path / f'{name}.tmp').open('wb') as file:
                np.savez(file, audio=chunk.audio, start=chunk.start_time, source=chunk.source)
                file.flush()
                os.fsync(file.fileno())
            (self.path / f'{name}.tmp').replace(self.path / f'{name}.npz')
            self.condition.notify()

    def get(self, timeout=0):
        from whisper_daemon.meeting_recorder import AudioChunk
        deadline = time.monotonic() + timeout
        with self.condition:
            while True:
                for path in sorted(self.path.glob('*.npz')):
                    if path.stem in self.inflight: continue
                    if path.with_suffix('.json').exists():
                        path.unlink()  # crash between durable result and unlink
                        continue
                    with np.load(path) as data:
                        chunk = AudioChunk(data['audio'], float(data['start']),
                                           len(data['audio']) / RATE, str(data['source']), path.stem)
                    self.inflight.add(path.stem)
                    return chunk
                remaining = deadline - time.monotonic()
                if remaining <= 0: raise queue.Empty
                self.condition.wait(remaining)

    def get_nowait(self):
        return self.get()

    def ack(self, chunk, result):
        result = {**result, 'source': chunk.source, 'duration': chunk.duration}
        with self.condition:
            atomic_json(self.path / f'{chunk.job_id}.json', [chunk.start_time, result])
            (self.path / f'{chunk.job_id}.npz').unlink(missing_ok=True)
            self.inflight.discard(chunk.job_id)

    def results(self):
        return [tuple(json.loads(p.read_text())) for p in sorted(self.path.glob('*.json'))]
