"""Private localhost meeting review with timestamp playback and durable edits."""
import copy
import json
import math
import re
import secrets
import shutil
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from whisper_daemon.formats import FORMATTERS
from whisper_daemon.meeting_manifest import publish_revision
from whisper_daemon.recording_store import atomic_json
from whisper_daemon import supervisor


def prepare_review(directory):
    path = Path(directory)
    transcript = json.loads((path / 'transcript.json').read_text())
    segments = transcript.get('segments', [])
    if not transcript.get('speakers'):
        transcript['speakers'] = [{'id': 0, 'label': 'Microphone'}, {'id': 1, 'label': 'Participants'}]
        for seg in segments: seg['speaker'] = 1 if seg.get('source') == 'browser' else 0
    for index, seg in enumerate(segments): seg['id'] = index
    notes_path = path / 'notes.json'
    if notes_path.exists():
        notes = json.loads(notes_path.read_text())
    else:
        important = [s for s in segments if re.search(r'\b(agreed|decided|decision|ustal\w*|decyz\w*|termin\w*)\b', s['text'], re.I)]
        tasks = [s for s in segments if re.search(r'\b(will|todo|action|wyśl\w*|przygotuj\w*|zrobi\w*|zadani\w*|trzeba)\b', s['text'], re.I)]
        notes = {'summary': '\n'.join(f"[{s['start']:.1f}s] {s['text'].strip()}" for s in (important or segments[:5])[:12]),
                 'tasks': [{'text': s['text'].strip(), 'start': s['start'], 'owner': '', 'due': '',
                            'done': False, 'confirmed': False} for s in tasks[:30]],
                 'method': 'Extractive suggestions; verify against the recording.'}
        atomic_json(notes_path, notes)
    session_path = path / 'session.json'
    session = json.loads(session_path.read_text()) if session_path.exists() else {}
    return {'title': path.name, 'transcript': transcript, 'notes': notes, 'session': session,
            'audio': [name for name in ('recording.wav', 'mic.wav', 'browser.wav') if (path / name).is_file()]}


def save_review(directory, changes):
    """Persist one complete review revision without racing a forced restart."""
    with supervisor.activity_transition():
        return _save_review(directory, changes)


def _save_review(directory, changes):
    if not isinstance(changes, dict): raise ValueError('Expected an object')
    path = Path(directory)
    data = prepare_review(path)
    transcript = copy.deepcopy(data['transcript'])
    names, assignments = changes.get('names', {}), changes.get('assignments', {})
    if not isinstance(names, dict) or not isinstance(assignments, dict): raise ValueError('Invalid speaker edits')
    known = {str(s['id']) for s in transcript['speakers']}
    for key in names:
        if key not in known:
            if not key.isdigit() or not 0 <= int(key) <= 1000 or len(transcript['speakers']) >= 50:
                raise ValueError('Invalid new speaker')
            transcript['speakers'].append({'id': int(key), 'label': names[key]})
    for speaker in transcript['speakers']:
        label = names.get(str(speaker['id']), speaker['label'])
        if not isinstance(label, str) or not label.strip() or len(label) > 80 or any(ord(c) < 32 for c in label):
            raise ValueError('Speaker names must be 1–80 printable characters')
        speaker['label'] = label.strip()
    ids = {s['id'] for s in transcript['speakers']}
    for segment in transcript['segments']:
        speaker = assignments.get(str(segment['id']), segment.get('speaker'))
        if speaker not in ids: raise ValueError('Unknown speaker')
        segment['speaker'] = speaker
    summary = changes.get('summary', data['notes']['summary'])
    tasks = changes.get('tasks', data['notes']['tasks'])
    if not isinstance(summary, str) or len(summary) > 50000: raise ValueError('Invalid summary')
    if not isinstance(tasks, list) or len(tasks) > 200: raise ValueError('Too many tasks')
    clean_tasks = []
    for task in tasks:
        if not isinstance(task, dict): raise ValueError('Invalid task')
        cleaned = {}
        for name in ('text', 'owner', 'due'):
            value = task.get(name, '')
            if not isinstance(value, str) or len(value) > 4000: raise ValueError('Invalid task text')
            cleaned[name] = value
        start = task.get('start', 0)
        if type(start) not in (int, float) or not math.isfinite(start) or start < 0: raise ValueError('Invalid timestamp')
        cleaned.update(start=start, done=task.get('done') is True, confirmed=task.get('confirmed') is True)
        clean_tasks.append(cleaned)
    original = path / 'transcript_original.json'
    if not original.exists(): shutil.copy2(path / 'transcript.json', original)
    atomic_json(path / 'transcript.json', transcript)
    atomic_json(path / 'notes.json', {'summary': summary, 'tasks': clean_tasks,
                                    'method': data['notes'].get('method', '')})
    for fmt in ('txt', 'srt', 'vtt'):
        target = path / f'transcript.{fmt}'
        temp = target.with_suffix(target.suffix + '.tmp')
        temp.write_text(FORMATTERS[fmt](transcript), encoding='utf-8')
        temp.replace(target)
    (path / 'notes.txt').write_text(summary + '\n\n' + '\n'.join(
        f"[{'x' if t['done'] else ' '}] {t['text']} | {t['owner']} | {t['due']} | {t['start']:.1f}s"
        for t in clean_tasks), encoding='utf-8')
    publish_revision(path)  # edits changed published files: Orbit needs a new revision
    return prepare_review(path)


class ReviewServer(ThreadingHTTPServer):
    daemon_threads = True
    def __init__(self, directory, port=0):
        self.directory = Path(directory).resolve()
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.RLock()
        prepare_review(self.directory)
        super().__init__(('127.0.0.1', port), ReviewHandler)
        self.url = f'http://127.0.0.1:{self.server_port}/{self.token}/'

    def start(self):
        threading.Thread(target=self.serve_forever, name='meeting-review', daemon=True).start()
        return self.url


class ReviewHandler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass  # URLs contain the private session token

    def _route(self):
        host = f'127.0.0.1:{self.server.server_port}'
        if self.headers.get('Host') != host: return None
        prefix = '/' + self.server.token + '/'
        return self.path[len(prefix):] if self.path.startswith(prefix) else None

    def _headers(self, status, kind, size):
        self.send_response(status)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(size))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; media-src 'self'; frame-ancestors 'none'")

    def _json(self, data, status=200):
        payload = json.dumps(data, ensure_ascii=False, allow_nan=False).encode()
        self._headers(status, 'application/json; charset=utf-8', len(payload))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        route = self._route()
        if route is None: self.send_error(403); return
        if route == 'data':
            with self.server.lock: self._json(prepare_review(self.server.directory))
            return
        if route in ('', 'review.js', 'review.css'):
            name = route or 'index.html'
            payload = (Path(__file__).parent / 'review_assets' / name).read_bytes()
            kind = {'index.html': 'text/html; charset=utf-8', 'review.js': 'text/javascript', 'review.css': 'text/css'}[name]
            self._headers(200, kind, len(payload)); self.end_headers(); self.wfile.write(payload)
            return
        if route not in ('recording.wav', 'mic.wav', 'browser.wav'): self.send_error(404); return
        path = self.server.directory / route
        if not path.is_file() or path.is_symlink(): self.send_error(404); return
        size = path.stat().st_size
        first, last, status = 0, size - 1, 200
        byte_range = self.headers.get('Range')
        if byte_range:
            match = re.fullmatch(r'bytes=(\d+)-(\d*)', byte_range)
            if not match: self.send_error(416); return
            first, last = int(match[1]), int(match[2]) if match[2] else last
            last = min(last, size - 1)
            if first > last: self.send_error(416); return
            status = 206
        self._headers(status, 'audio/wav', last - first + 1)
        self.send_header('Accept-Ranges', 'bytes')
        if status == 206: self.send_header('Content-Range', f'bytes {first}-{last}/{size}')
        self.end_headers()
        try:
            with path.open('rb') as audio:
                audio.seek(first)
                remaining = last - first + 1
                while remaining:
                    chunk = audio.read(min(65536, remaining))
                    if not chunk: break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError): pass

    def do_POST(self):
        if self._route() != 'save': self.send_error(403); return
        expected = f'http://127.0.0.1:{self.server.server_port}'
        if self.headers.get('Origin') != expected: self.send_error(403); return
        try:
            size = int(self.headers.get('Content-Length', 0))
            if not 0 < size <= 1_000_000: raise ValueError('Invalid request size')
            data = json.loads(self.rfile.read(size))
            with self.server.lock: result = save_review(self.server.directory, data)
            self._json(result)
        except (ValueError, TypeError, KeyError) as exc:
            self._json({'error': str(exc)}, 400)
