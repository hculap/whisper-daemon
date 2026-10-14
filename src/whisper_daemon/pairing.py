"""Explicit local pairing with one Chrome extension installation."""
import json
import os
import re
import secrets
from pathlib import Path
from whisper_daemon.config import CONFIG_DIR

PAIRING_FILE = CONFIG_DIR / 'bridge-pairing.json'


def load_pairing(path=None):
    try:
        data = json.loads(Path(path or PAIRING_FILE).read_text())
        if not re.fullmatch(r'[a-p]{32}', data['extension_id']): return {}
        if len(data['secret']) < 32: return {}
        return data
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def pair(extension_id, path=None):
    if not re.fullmatch(r'[a-p]{32}', extension_id):
        raise ValueError('Extension ID must be the 32 letters shown in chrome://extensions')
    target = Path(path or PAIRING_FILE)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = {'extension_id': extension_id, 'secret': secrets.token_urlsafe(32)}
    temp = target.with_suffix('.tmp')
    with open(temp, 'w', opener=lambda p, flags: os.open(p, flags, 0o600)) as file:
        os.chmod(temp, 0o600)
        json.dump(data, file)
        file.flush()
        os.fsync(file.fileno())
    temp.replace(target)
    return data
