"""Exercise worker lifecycle guards without starting Cocoa or real devices."""
import ast
from pathlib import Path
from types import SimpleNamespace


def method(name, **globals):
    tree = ast.parse(Path('src/whisper_daemon/menubar.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MenuBarDelegate')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    space = {'logger': SimpleNamespace(info=lambda *args: None)}
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
        _browser_audio_lock=threading.RLock(),
        _start_meeting=lambda *args:starts.append(args))
    helper=SimpleNamespace(callAfter=lambda fn,*args:queue.append(lambda:fn(*args)))
    method('_on_browser_connect',AppHelper=helper)(obj,'test','',bridge.session_id,100.)
    bridge.session_id=None
    for callback in queue: callback()
    assert not starts


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
