import queue
from types import SimpleNamespace
import numpy as np
from whisper_daemon import daemon as mod
from whisper_daemon.events import Event, EventType
from whisper_daemon.history import HistoryStore


def fixture(tmp_path):
    rec = SimpleNamespace(stop_recording=lambda: np.zeros(0), voice_detected=True)
    settings = SimpleNamespace(dictation_confirm=False)
    return mod.Daemon(queue.Queue(), rec, settings=settings, history_store=HistoryStore(tmp_path / 'history.json'))


def test_cancelled_result_cannot_paste_into_new_dictation(tmp_path, monkeypatch):
    daemon = fixture(tmp_path)
    daemon._state = mod.State.TRANSCRIBING
    daemon._recording_gen = 1
    daemon._handle_event(Event(EventType.RECORD_CANCEL))
    daemon._recording_gen = 3
    daemon._state = mod.State.TRANSCRIBING
    pasted = []
    monkeypatch.setattr(mod, 'paste_text', lambda *args, **kw: pasted.append(args))
    daemon._handle_transcription_done({'generation': 1, 'text': 'cancelled'})
    assert pasted == [] and daemon.history == []


def test_changed_focus_keeps_result_in_persistent_history(tmp_path, monkeypatch):
    daemon = fixture(tmp_path)
    daemon._state = mod.State.TRANSCRIBING
    daemon._paste_target = ('original',)
    monkeypatch.setattr(mod, 'paste_text', lambda *args, **kwargs: False)
    monkeypatch.setattr(mod, 'notify', lambda *args: None)
    daemon._handle_transcription_done({'generation': 0, 'text': 'keep this result'})
    assert daemon.history == ['keep this result']
    assert HistoryStore(tmp_path / 'history.json').texts() == ['keep this result']


def test_confirmation_requires_explicit_accept(tmp_path, monkeypatch):
    daemon = fixture(tmp_path)
    daemon._settings.dictation_confirm = True
    daemon._state = mod.State.TRANSCRIBING
    calls = []
    monkeypatch.setattr(mod, 'paste_text', lambda *args, **kw: calls.append(args) or True)
    daemon._paste_target = ('target',)
    daemon._handle_transcription_done({'generation': 0, 'text': 'draft'})
    assert daemon._state == mod.State.REVIEW and not calls
    daemon._handle_event(Event(EventType.DICTATION_ACCEPT, 'edited draft'))
    assert calls[0][0] == 'edited draft'


def test_history_retention_is_bounded(tmp_path):
    store = HistoryStore(tmp_path / 'history.json', limit=3)
    for index in range(5): store.add(str(index))
    assert HistoryStore(tmp_path / 'history.json').texts() == ['4', '3', '2']
