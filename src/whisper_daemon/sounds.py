"""Audio feedback using macOS system sounds."""

import logging
import threading

from AppKit import NSSound

logger = logging.getLogger(__name__)

# macOS built-in sounds (in /System/Library/Sounds/)
START_SOUND = "Tink"
STOP_SOUND = "Pop"
ERROR_SOUND = "Basso"


def _play_async(name: str) -> None:
    """Play a macOS system sound without blocking."""
    sound = NSSound.soundNamed_(name)
    if sound is None:
        logger.warning("System sound '%s' not found", name)
        return
    sound.play()


def play_start() -> None:
    threading.Thread(target=_play_async, args=(START_SOUND,), daemon=True).start()


def play_stop() -> None:
    threading.Thread(target=_play_async, args=(STOP_SOUND,), daemon=True).start()


def play_error() -> None:
    threading.Thread(target=_play_async, args=(ERROR_SOUND,), daemon=True).start()
