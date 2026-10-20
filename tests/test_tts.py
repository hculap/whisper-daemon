import threading

from whisper_daemon import tts


def test_speaking_is_reported_from_request_until_playback_ends(monkeypatch):
    playing = threading.Event()
    finish = threading.Event()
    calls = []

    def run(args, **kwargs):
        calls.append(args[0])
        if args[0] == "afplay":
            playing.set()
            assert finish.wait(2)
        return type("Done", (), {"returncode": 0, "stderr": ""})()

    started = []
    real_thread = threading.Thread

    class DeferredThread(real_thread):
        def start(self):
            # The busy flag must already be up before the worker gets to run.
            started.append(tts.is_speaking())
            super().start()

    monkeypatch.setattr(tts.subprocess, "run", run)
    monkeypatch.setattr(tts.threading, "Thread", DeferredThread)
    monkeypatch.setattr(tts, "PIPER_BIN", type("Path", (), {"exists": lambda self: True})())
    monkeypatch.setitem(tts.VOICES, "en", type("Path", (), {"exists": lambda self: True, "name": "en"})())

    assert not tts.is_speaking()
    tts.speak_text("Hello there", language="en")
    assert started == [True]
    assert playing.wait(2)
    assert tts.is_speaking()
    tts.speak_text("Second request", language="en")  # skipped while speaking
    assert started == [True]
    finish.set()
    for _ in range(200):
        if not tts.is_speaking():
            break
        threading.Event().wait(0.01)
    assert not tts.is_speaking()
    assert calls == [str(tts.PIPER_BIN), "afplay"]
