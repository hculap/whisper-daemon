"""Versioned, atomic readiness contract for consumers of synced recordings (stdlib only)."""
from __future__ import annotations

import concurrent.futures
from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
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


# Never published: the markers themselves, the live partial, and the recovery-only
# internals of RecordingStore/DiskChunkQueue (raw float32 sources and the chunk
# spool). The aligned WAV exports are the published audio.
UNPUBLISHED_NAMES = ('meeting.ready.json', 'meeting.in-progress.json', 'meeting.failed.json', 'transcript_live.txt')
UNPUBLISHED_SUFFIXES = ('.tmp', '.f32')
UNPUBLISHED_DIRS = ('chunks',)


def begin_meeting(folder: Path, title: str, *, started_at=None, timezone_name=None):
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
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


def finalize_meeting(folder: Path, identity, result, errors, *, target: Path | None = None):
    """Publish meeting.ready.json last, or meeting.failed.json if anything is missing.

    ``result=None`` means the pipeline already wrote the final transcript.json.
    With ``target``, ``folder`` is a local capture folder: only the published
    files are copied to the new ``target`` and ``folder`` (raw sources, job
    spool) is deleted afterwards. A failed meeting stays in ``folder``.
    """
    if (folder / 'meeting.ready.json').exists():
        raise FileExistsError('Meeting already finalized; never change a published revision')
    if result is None:
        result = json.loads((folder / 'transcript.json').read_text(encoding='utf-8'))
    else:
        # Always export machine-readable transcription, irrespective of display format preferences.
        atomic_json(folder / 'transcript.json', result)
    failures = list(errors)
    if not result.get('text', '').strip() and not result.get('segments'):
        failures.append('No speech transcribed')
    # DiskChunkQueue deletes a job's audio only once its result is durable.
    pending = list((folder / 'chunks').glob('*.npz'))
    if pending:
        failures.append(f'{len(pending)} transcription job(s) never completed')
    if failures:
        fail_meeting(folder, identity, failures)
        return None
    if target is None or Path(target).resolve() == folder.resolve():
        published = folder
    else:
        published = Path(target)
        _copy_published(folder, published)
    manifest = dict(identity, transcription_complete=True, transcription_errors=[], files=_published_files(published))
    atomic_json(published / 'meeting.ready.json', manifest)  # last published file
    if published == folder:
        (folder / 'meeting.in-progress.json').unlink(missing_ok=True)
        (folder / 'meeting.failed.json').unlink(missing_ok=True)
    else:
        try:
            shutil.rmtree(folder)
        except OSError:
            logging.getLogger(__name__).warning('Published %s, but could not remove the capture folder %s', published, folder)
    return manifest


def publish_revision(folder: Path):
    """Republish a ready meeting whose files changed (review edits) as the next revision.

    A published revision is immutable, so a changed file needs a new one. A no-op
    for meetings that were never ready and for saves that changed nothing.
    """
    marker = folder / 'meeting.ready.json'
    if not marker.exists():
        return None
    current = json.loads(marker.read_text(encoding='utf-8'))
    files = _published_files(folder)
    if files == current['files']:
        return current
    manifest = dict(current, revision=current['revision'] + 1, files=files)
    atomic_json(marker, manifest)
    return manifest


def fail_meeting(folder: Path, identity, errors):
    atomic_json(folder / 'meeting.failed.json',
                dict(identity, transcription_complete=False, transcription_errors=list(errors)))


def _publishable(folder: Path):
    for path in sorted(folder.rglob('*')):
        if path.is_symlink():
            raise ValueError('Symlinks cannot be published in a recording')
        parts = path.relative_to(folder).parts
        if not path.is_file() or any(part.startswith('.') for part in parts) or parts[0] in UNPUBLISHED_DIRS:
            continue
        if path.name in UNPUBLISHED_NAMES or path.name.endswith(UNPUBLISHED_SUFFIXES):
            continue
        yield path


def _copy_published(folder: Path, target: Path):
    """Copy the published files into a new ``target``; a partial copy is removed."""
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(mode=0o700)  # never merge into an existing meeting
    try:
        for path in _publishable(folder):
            destination = target / path.relative_to(folder)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            with destination.open('rb') as stream:
                os.fsync(stream.fileno())
    except BaseException:
        shutil.rmtree(target, ignore_errors=True)
        raise


def _published_files(folder: Path):
    files = []
    for path in _publishable(folder):
        digest = hashlib.sha256()
        count = 0
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                count += len(chunk)
                digest.update(chunk)
        relative = path.relative_to(folder).as_posix()
        role = 'transcript' if relative == 'transcript.json' else ('audio' if path.suffix.lower() in ('.wav', '.mp3', '.m4a', '.flac') else 'attachment')
        files.append({'path': relative, 'size': count, 'sha256': digest.hexdigest(), 'role': role})
    return files


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
            logging.getLogger(__name__).error('Chunk transcription failed: %s (%s)', message, exc)
    return collected
