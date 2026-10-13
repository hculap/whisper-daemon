"""Versioned, atomic readiness contract for consumers of synced recordings (stdlib only)."""
from __future__ import annotations

import concurrent.futures
from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile
from uuid import uuid4
from zoneinfo import ZoneInfo


def atomic_json(path: Path, value):
    fd, name = tempfile.mkstemp(prefix='.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        Path(name).unlink(missing_ok=True)


def local_timezone():
    candidates = [os.environ.get('TZ', '')]
    try:
        candidates.append(str(Path('/etc/localtime').resolve()).split('/zoneinfo/', 1)[1])
    except (OSError, IndexError):
        pass
    for value in candidates:
        try:
            ZoneInfo(value)
            return value
        except (ValueError, KeyError):
            pass
    return 'UTC'


def begin_meeting(folder: Path, title: str, *, started_at=None, timezone_name=None):
    folder.mkdir(parents=True, exist_ok=True)
    for name in ('meeting.ready.json', 'meeting.in-progress.json', 'meeting.failed.json'):
        if (folder / name).exists():
            raise FileExistsError('Use a new recording directory; meeting identity is immutable')
    timezone_name = timezone_name or local_timezone()
    ZoneInfo(timezone_name)
    started_at = started_at or datetime.now().astimezone()
    if started_at.utcoffset() is None:
        raise ValueError('started_at requires an offset')
    value = {'schema_version': 1, 'meeting_id': str(uuid4()), 'revision': 1,
             'title': (title or 'Meeting')[:300], 'started_at': started_at.isoformat(), 'timezone': timezone_name}
    atomic_json(folder / 'meeting.in-progress.json', value)
    return value


def finalize_meeting(folder: Path, identity, result, errors):
    if (folder / 'meeting.ready.json').exists():
        raise FileExistsError('Meeting already finalized; never change a published revision')
    # Always export machine-readable transcription, irrespective of display format preferences.
    atomic_json(folder / 'transcript.json', result)
    failures = list(errors)
    if not result.get('text', '').strip() and not result.get('segments'):
        failures.append('No speech transcribed')
    if failures:
        atomic_json(folder / 'meeting.failed.json', dict(identity, transcription_complete=False, transcription_errors=failures))
        return None
    files = []
    for path in sorted(folder.rglob('*')):
        if path.is_symlink():
            raise ValueError('Symlinks cannot be published in a recording')
        relative = path.relative_to(folder).as_posix()
        if not path.is_file() or any(part.startswith('.') for part in path.relative_to(folder).parts):
            continue
        if path.name in ('meeting.in-progress.json', 'meeting.failed.json', 'transcript_live.txt') or path.name.endswith('.tmp'):
            continue
        digest = hashlib.sha256()
        count = 0
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                count += len(chunk)
                digest.update(chunk)
        role = 'transcript' if relative == 'transcript.json' else ('audio' if path.suffix.lower() in ('.wav', '.mp3', '.m4a', '.flac') else 'attachment')
        files.append({'path': relative, 'size': count, 'sha256': digest.hexdigest(), 'role': role})
    manifest = dict(identity, transcription_complete=True, transcription_errors=[], files=files)
    atomic_json(folder / 'meeting.ready.json', manifest)  # last published file
    (folder / 'meeting.in-progress.json').unlink(missing_ok=True)
    (folder / 'meeting.failed.json').unlink(missing_ok=True)
    return manifest


def collect_futures(futures, all_results, errors, *, wait=False):
    done = concurrent.futures.wait(futures)[0] if wait and futures else {f for f in futures if f.done()}
    collected = False
    for future in done:
        start_time = futures.pop(future)
        try:
            result = future.result()
            if isinstance(result, str):
                result = {'text': result, 'segments': [], 'language': ''}
            if result.get('text', '').strip():
                all_results.append((start_time, result))
                collected = True
        except Exception as exc:
            message = f'Chunk at {start_time:.3f}s: {type(exc).__name__}'
            errors.append(message)
            logging.getLogger(__name__).error(message)
    return collected
