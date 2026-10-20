"""Exercise worker lifecycle guards without starting Cocoa or real devices."""
import ast
from pathlib import Path
from types import SimpleNamespace


def method(name, **globals):
    from whisper_daemon import supervisor

    tree = ast.parse(Path('src/whisper_daemon/menubar.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MenuBarDelegate')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    space = {
        'logger': SimpleNamespace(info=lambda *args: None),
        'supervisor': supervisor,
    }
    space.update(globals)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), 'menubar.py', 'exec'), space)
    return space[name]


def test_start_during_finalization_cannot_replace_worker():
    running = SimpleNamespace(is_alive=lambda: True)
    obj = SimpleNamespace(_meeting_active=False, _meeting_phase='transcribing', _meeting_thread=running)
    method('_start_meeting')(obj)
    assert obj._meeting_thread is running
    assert obj._meeting_phase == 'transcribing'


def test_file_job_finishing_does_not_reset_a_meeting():
    obj = SimpleNamespace(_meeting_active=True, _meeting_phase='recording')
    method('_finish_file_ui')(obj)
    assert obj._meeting_active and obj._meeting_phase == 'recording'


def test_pause_before_store_is_ready_survives_handoff(tmp_path):
    import threading
    import numpy as np
    from whisper_daemon.recording_store import RecordingStore
    obj = SimpleNamespace(_meeting_active=True, _meeting_paused=False, _meeting_store=None,
        _meeting_start=100, _meeting_pause_events=[], _browser_audio_lock=threading.RLock(),
        _set_status_safe=lambda *args: None)
    method('onPauseMeeting_', time=SimpleNamespace(monotonic=lambda:102))(obj, None)
    store = RecordingStore(tmp_path, clock=lambda:103, started_at=100)
    method('_attach_recording_store')(obj, store)
    assert obj._meeting_paused
    assert store.meta['pauses'] == [{'time':2., 'paused':True}]
    assert store.unpaused_audio(np.ones(512),32000) == []


def test_cancel_before_cocoa_dispatch_cannot_start_microphone():
    import threading
    queue, starts = [], []
    bridge=SimpleNamespace(session_id='session-12345678')
    obj=SimpleNamespace(_meeting_phase='idle',_meeting_thread=None,_browser_bridge=bridge,
        _browser_audio_lock=threading.RLock(),_browser_start_pending=None,
        _start_meeting=lambda *args:starts.append(args))
    helper=SimpleNamespace(callAfter=lambda fn,*args:queue.append(lambda:fn(*args)))
    method('_on_browser_connect',AppHelper=helper)(obj,'test','',bridge.session_id,100.)
    bridge.session_id=None
    for callback in queue: callback()
    assert not starts


def test_stop_before_cocoa_dispatch_saves_staged_audio_without_devices():
    import threading
    queue, calls = [], []
    bridge=SimpleNamespace(session_id='session-12345678')
    obj=SimpleNamespace(_meeting_phase='idle',_meeting_thread=None,_browser_bridge=bridge,
        _browser_audio_lock=threading.RLock(),_browser_start_pending=None,
        _start_meeting=lambda *args,**kwargs:calls.append(('start',args,kwargs)),
        _stop_meeting=lambda:calls.append(('stop',)))
    helper=SimpleNamespace(callAfter=lambda fn,*args:queue.append(lambda:fn(*args)))
    connect=method('_on_browser_connect',AppHelper=helper)
    connect(obj,'test','',bridge.session_id,100.)
    obj._browser_prebuffer=[(b'tail',0)]  # a recovered tail: start, PCM, stop
    bridge.session_id=None
    assert connect(obj,'next','','session-87654321',101.) is False  # the tail owns the staging
    for callback in queue: callback()
    assert calls == [('start',(True,'test',100.,'session-12345678'),{'browser_only':True}),('stop',)]
    assert obj._browser_start_pending is None


def test_stop_during_worker_startup_preserves_acknowledged_pcm():
    import threading
    obj = SimpleNamespace(_meeting_active=True, _meeting_browser_triggered=True,
        _meeting_session_id='test-session', _meeting_phase='recording',
        _browser_audio_lock=threading.RLock(), _browser_prebuffer=[(b'audio',0)],
        _browser_prebuffer_bytes=5)
    obj._stop_meeting=lambda:setattr(obj,'_meeting_active',False)
    helper=SimpleNamespace(callAfter=lambda fn,*args:fn(*args))
    method('_on_browser_disconnect',AppHelper=helper)(obj, 'test-session')
    assert not obj._meeting_active
    assert obj._browser_prebuffer == [(b'audio',0)]


def test_stop_closes_microphone_while_speaker_model_probe_is_still_blocked(tmp_path):
    """Model warmup may delay finalization, but it must never extend capture."""
    import concurrent.futures, datetime, logging, queue, re, sys, threading, time, types
    from dataclasses import dataclass
    import numpy as np
    from whisper_daemon.bridge_protocol import build_status_message
    from whisper_daemon.config import Settings

    probe_started = threading.Event()
    release_probe = threading.Event()
    mic_started = threading.Event()
    mic_stopped = threading.Event()

    def diarization_status(**kwargs):
        probe_started.set()
        assert release_probe.wait(5)
        return {'available': False, 'message': 'test model unavailable'}

    @dataclass(frozen=True)
    class AudioChunk:
        audio: np.ndarray
        start_time: float
        duration: float
        source: str = 'mic'
        job_id: str = ''

    class Recorder:
        error, needs_recovery, _source_label = None, False, 'mic'
        def __init__(self, *args, **kwargs): pass
        def start(self): mic_started.set()
        def stop(self): mic_stopped.set()
        def abort(self): pass
        def check_liveness(self, now): return False

    modules = {
        'whisper_daemon.meeting_recorder': types.SimpleNamespace(MeetingRecorder=Recorder, AudioChunk=AudioChunk),
        'whisper_daemon.screen_capture': types.SimpleNamespace(ScreenCapture=None),
        'whisper_daemon.transcriber': types.SimpleNamespace(transcribe_full=lambda *args: None),
        'whisper_daemon.model_policy': types.SimpleNamespace(diarization_status=diarization_status),
    }
    tree = ast.parse(Path('src/whisper_daemon/menubar.py').read_text())
    write_partial = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_write_partial')
    space = {'Path': Path}
    exec(compile(ast.Module(body=[write_partial], type_ignores=[]), 'menubar.py', 'exec'), space)
    worker = method('_meeting_worker', re=re, datetime=datetime.datetime, timedelta=datetime.timedelta,
        concurrent=concurrent, np=np, time=time, queue=queue, logger=logging.getLogger('test'),
        _notify=lambda *args: None, _write_partial=space['_write_partial'],
        build_status_message=build_status_message)
    now = time.monotonic()
    obj = SimpleNamespace(_settings=Settings(recording_dir=str(tmp_path), capture_mic=True,
            capture_tab=False, capture_screenshots=False, diarize=True),
        _daemon=SimpleNamespace(_model='model'), _meeting_active=True, _meeting_paused=False,
        _meeting_start=now, _meeting_stopped_at=None, _meeting_session_id='session-12345678',
        _browser_bridge=SimpleNamespace(send_json_str=lambda text: None), _browser_audio_lock=threading.RLock(),
        _browser_prebuffer=[], _browser_prebuffer_bytes=0, _browser_recorder=None,
        _meeting_store=None, _last_meeting_dir=None, _diarization_status=None,
        _meeting_phase='recording', _meeting_chunk_count=0,
        _attach_recording_store=lambda store: None, _send_results_to_browser=lambda *args: None,
        _reset_meeting_ui=lambda: None)
    real_modules = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    thread = threading.Thread(target=worker, args=(obj, False, 'Daily', False))
    try:
        thread.start()
        assert mic_started.wait(2)
        assert probe_started.wait(2)
        obj._meeting_active = False
        obj._meeting_stopped_at = time.monotonic()
        assert mic_stopped.wait(1), 'Stop waited for the speaker-model probe'
    finally:
        release_probe.set()
        thread.join(5)
        for name, module in real_modules.items():
            if module is None: sys.modules.pop(name, None)
            else: sys.modules[name] = module
    assert not thread.is_alive()


def run_worker(tmp_path, transcribe):
    """The real _meeting_worker for a staged browser tail; Whisper/devices faked."""
    import concurrent.futures, datetime, logging, queue, re, sys, threading, time, types
    from dataclasses import dataclass
    import numpy as np
    from whisper_daemon.bridge_protocol import build_status_message
    from whisper_daemon.config import Settings

    @dataclass(frozen=True)
    class AudioChunk:  # meeting_recorder.AudioChunk; that module needs PortAudio
        audio: np.ndarray
        start_time: float
        duration: float
        source: str = 'mic'
        job_id: str = ''

    class Recorder:  # stores PCM before queueing a job, like MeetingRecorder
        error, needs_recovery = None, False
        def __init__(self, jobs, source_label='mic', raw_sink=None, **kwargs):
            self.jobs, self._source_label, self.raw_sink = jobs, source_label, raw_sink
        def start_without_device(self): pass
        def feed_audio(self, samples, position=None):
            start = self.raw_sink(samples, position)
            self.jobs.put(AudioChunk(samples, start, len(samples) / 16000, self._source_label))
        def stop_without_device(self): pass
        def abort(self): pass

    modules = {'whisper_daemon.meeting_recorder': types.SimpleNamespace(MeetingRecorder=Recorder, AudioChunk=AudioChunk),
               'whisper_daemon.screen_capture': types.SimpleNamespace(ScreenCapture=None),
               'whisper_daemon.transcriber': types.SimpleNamespace(transcribe_full=transcribe)}
    tree = ast.parse(Path('src/whisper_daemon/menubar.py').read_text())
    write_partial = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_write_partial')
    space = {'Path': Path}
    exec(compile(ast.Module(body=[write_partial], type_ignores=[]), 'menubar.py', 'exec'), space)
    notes = []
    worker = method('_meeting_worker', re=re, datetime=datetime.datetime, timedelta=datetime.timedelta,
        concurrent=concurrent, np=np, time=time, queue=queue, logger=logging.getLogger('test'),
        _notify=lambda *parts: notes.append(parts), _write_partial=space['_write_partial'],
        build_status_message=build_status_message)
    now = time.monotonic()
    obj = SimpleNamespace(_settings=Settings(recording_dir=str(tmp_path), capture_mic=False, capture_screenshots=False),
        _daemon=SimpleNamespace(_model='model'), _meeting_active=False, _meeting_paused=False,
        _meeting_start=now - 5, _meeting_stopped_at=now, _meeting_session_id='session-12345678',
        _browser_bridge=SimpleNamespace(send_json_str=lambda text: None), _browser_audio_lock=threading.RLock(),
        _browser_prebuffer=[(np.full(16000, .1, dtype='<f4').tobytes(), 0)], _browser_prebuffer_bytes=64000,
        _browser_recorder=None, _meeting_store=None, _last_meeting_dir=None, _diarization_status=None,
        _meeting_phase='transcribing', _meeting_chunk_count=0, _attach_recording_store=lambda store: None,
        _send_results_to_browser=lambda *args: None, _reset_meeting_ui=lambda: None)
    real_modules = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        worker(obj, True, 'CRM review', True)  # browser_only: a stopped, recovered tail
    finally:
        for name, module in real_modules.items():
            if module is None: sys.modules.pop(name, None)
            else: sys.modules[name] = module
    return next(tmp_path.glob('recording_*')), notes


def test_finished_meeting_publishes_ready_manifest_last(tmp_path):
    import hashlib, json, datetime
    folder, notes = run_worker(tmp_path, lambda audio, model, language: {
        'text': 'Szymon wyśle ofertę.', 'language': 'pl',
        'segments': [{'start': 0, 'end': 1, 'text': 'Szymon wyśle ofertę.'}]})
    manifest = json.loads((folder / 'meeting.ready.json').read_text())
    paths = {f['path']: f for f in manifest['files']}
    assert {'transcript.json', 'transcript.txt', 'notes.json', 'session.json', 'recording.wav', 'browser.wav'} <= set(paths)
    assert not [p for p in paths if p.startswith('chunks/') or p.endswith('.f32') or p.startswith('meeting.')]
    assert paths['transcript.json']['role'] == 'transcript' and paths['browser.wav']['role'] == 'audio'
    for entry in manifest['files']:
        content = (folder / entry['path']).read_bytes()
        assert entry['sha256'] == hashlib.sha256(content).hexdigest() and entry['size'] == len(content)
        assert (folder / entry['path']).stat().st_mtime_ns <= (folder / 'meeting.ready.json').stat().st_mtime_ns
    assert json.loads((folder / 'session.json').read_text())['state'] == 'complete'
    assert json.loads((folder / 'transcript.json').read_text())['segments'][0]['source'] == 'browser'
    assert manifest['title'] == 'CRM review' and manifest['transcription_complete']
    started = datetime.datetime.fromisoformat(manifest['started_at'])
    assert 4 < (datetime.datetime.now().astimezone() - started).total_seconds() < 60  # meeting, not worker, start
    assert not (folder / 'meeting.in-progress.json').exists() and not (folder / 'meeting.failed.json').exists()
    assert [n[1] for n in notes] == ['Meeting saved']


def test_failed_chunk_marks_meeting_failed_and_never_ready(tmp_path):
    import json
    def broken(audio, model, language):
        raise RuntimeError('decoder failed')
    folder, notes = run_worker(tmp_path, broken)
    assert not (folder / 'meeting.ready.json').exists()
    failed = json.loads((folder / 'meeting.failed.json').read_text())
    assert failed['transcription_errors'] == ['RuntimeError: decoder failed']
    assert list((folder / 'chunks').glob('*.npz'))  # the job is still recoverable
    assert [n[1] for n in notes] == ['Meeting interrupted']


def test_frame_racing_the_session_end_is_not_stranded_in_staging():
    import threading
    bridge = SimpleNamespace(session_id='session-12345678', send_json_str=lambda text: None)
    obj = SimpleNamespace(_browser_recorder=None, _meeting_phase='idle', _browser_bridge=bridge,
        _browser_audio_lock=threading.RLock(), _browser_prebuffer=[], _browser_prebuffer_bytes=0)
    on_audio = method('_on_browser_audio', BROWSER_PREBUFFER_MAX_BYTES=1 << 20)
    on_audio(obj, b'\0' * 8, 0)  # a pending start owns this startup tail
    assert obj._browser_prebuffer_bytes == 8
    obj._browser_prebuffer, obj._browser_prebuffer_bytes = [], 0
    bridge.session_id = None  # _reset_meeting_ui ended the session first
    on_audio(obj, b'\0' * 8, 2)
    assert obj._browser_prebuffer == [] and obj._browser_prebuffer_bytes == 0


def test_format_toggle_returns_a_new_list_and_never_empties():
    tree = ast.parse(Path('src/whisper_daemon/menubar.py').read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_toggle_format')
    space = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), 'menubar.py', 'exec'), space)
    toggle = space['_toggle_format']
    formats = ['txt']
    assert toggle(formats, 'srt') == ['txt', 'srt']
    assert toggle(['txt', 'srt'], 'txt') == ['srt']
    kept = toggle(formats, 'txt')
    assert kept == ['txt'] and kept is not formats
    assert formats == ['txt']


def test_diarized_file_batch_keeps_the_pipeline_until_the_end(tmp_path, monkeypatch):
    from whisper_daemon import transcriber
    files = []
    for name in ('a.wav', 'b.wav', 'c.wav'):
        (tmp_path / name).write_bytes(b'')
        files.append(str(tmp_path / name))
    calls = []
    monkeypatch.setattr(transcriber, 'transcribe_file', lambda path, model: {'text': path, 'segments': []})

    def diarize(path, result, *, release=True):
        calls.append(('diarize', release))
        return result

    settings = SimpleNamespace(transcription_diarize=True, transcription_output_dir_path=tmp_path / 'out',
        transcription_formats=['txt'])
    obj = SimpleNamespace(_daemon=SimpleNamespace(_model='test'), _settings=settings, _meeting_phase='idle',
        _set_icon_safe=lambda state: None, _set_status_safe=lambda text: None,
        _finish_file_ui=lambda: calls.append('finish'))
    worker = method('_transcribe_paths_worker', Path=Path, State=SimpleNamespace(TRANSCRIBING='transcribing'),
        AUDIO_VIDEO_EXTENSIONS={'wav'}, _notify=lambda *args: None, _diarize_transcript=diarize,
        _release_diarizer=lambda: calls.append('release'))
    worker(obj, files)
    assert calls == [('diarize', False)] * 3 + ['release', 'finish']
