"""Bounded local dictation history, including results held back by focus protection."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from whisper_daemon.config import CONFIG_DIR
from whisper_daemon.recording_store import atomic_json

class HistoryStore:
    def __init__(self, path=None, limit=100):
        self.path = Path(path or CONFIG_DIR / 'dictation-history.json')
        self.limit = limit
        try:
            entries = json.loads(self.path.read_text())
            self.entries = [e for e in entries if isinstance(e, dict) and isinstance(e.get('text'), str)][:limit]
        except (OSError, ValueError, TypeError):
            self.entries = []

    def texts(self):
        return [entry['text'] for entry in self.entries]

    def add(self, text):
        self.entries.insert(0, {'text': text, 'created_at': datetime.now(timezone.utc).isoformat()})
        del self.entries[self.limit:]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(self.path, self.entries)
        os.chmod(self.path, 0o600)

    def clear(self):
        self.entries = []
        if self.path.exists(): atomic_json(self.path, [])
