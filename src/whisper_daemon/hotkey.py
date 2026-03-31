"""Global hotkey listener using pynput."""

import logging
import queue

from pynput import keyboard

from whisper_daemon.events import Event, EventType

logger = logging.getLogger(__name__)

HOTKEY_COMBO = "<cmd>+<shift>+<space>"


class HotkeyListener:
    """Listens for a global hotkey and emits RECORD_TOGGLE events."""

    def __init__(self, event_queue: queue.Queue[Event]) -> None:
        self._queue = event_queue
        self._listener = keyboard.GlobalHotKeys({
            HOTKEY_COMBO: self._on_activate,
        })
        self._listener.daemon = True

    def _on_activate(self) -> None:
        logger.info("Hotkey pressed")
        self._queue.put(Event(EventType.RECORD_TOGGLE))

    def start(self) -> None:
        logger.info("Hotkey listener started (%s)", HOTKEY_COMBO)
        self._listener.start()

    def stop(self) -> None:
        self._listener.stop()
        logger.info("Hotkey listener stopped")
