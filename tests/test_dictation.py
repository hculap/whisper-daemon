import queue
from types import SimpleNamespace

import numpy as np

from whisper_daemon import daemon as mod


def test_final_transcription_includes_audio_after_preview(monkeypatch):
    audio = np.ones(160000, dtype=np.float32)
    rec = SimpleNamespace(stop_recording=lambda: audio, voice_detected=True)
    daemon = mod.Daemon(queue.Queue(), rec)
    daemon._pending_text = "preview only"
    daemon._pending_samples = 128000
    seen = []
    daemon._transcribe_worker = lambda final: seen.append(final)
    class InlineThread:
        def __init__(self, target, args, **kwargs): self.target, self.args = target, args
        def start(self): self.target(*self.args)
    monkeypatch.setattr(mod.threading, "Thread", InlineThread)
    monkeypatch.setattr(mod, "play_stop", lambda: None)
    daemon._start_transcription()
    assert len(seen) == 1
    assert len(seen[0]) == 160000
