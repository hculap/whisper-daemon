"""Exercise worker lifecycle guards without starting Cocoa or real devices."""
import ast
from pathlib import Path
from types import SimpleNamespace


def method(name):
    tree = ast.parse(Path('src/whisper_daemon/menubar.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MenuBarDelegate')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    space = {'logger': SimpleNamespace(info=lambda *args: None)}
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
