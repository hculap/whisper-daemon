"""Global hotkey listener using pynput."""

import logging
import queue

from pynput import keyboard

from whisper_daemon.events import Event, EventType

logger = logging.getLogger(__name__)

RECORD_HOTKEY = "<cmd>+<shift>+<space>"
PASTE_LAST_HOTKEY = "<cmd>+<shift>+v"


class HotkeyListener:
    """Listens for global hotkeys: record toggle and paste-last."""

    def __init__(self, event_queue: queue.Queue[Event]) -> None:
        self._queue = event_queue
        self._listener = keyboard.GlobalHotKeys({
            RECORD_HOTKEY: self._on_record,
            PASTE_LAST_HOTKEY: self._on_paste_last,
        })
        self._listener.daemon = True

    def _on_record(self) -> None:
        logger.info("Hotkey: record toggle")
        self._queue.put(Event(EventType.RECORD_TOGGLE))

    def _on_paste_last(self) -> None:
        logger.info("Hotkey: paste last transcription")
        self._queue.put(Event(EventType.PASTE_LAST))

    def start(self) -> None:
        logger.info("Hotkeys: %s (record), %s (paste last)", RECORD_HOTKEY, PASTE_LAST_HOTKEY)
        self._listener.start()

    def stop(self) -> None:
        self._listener.stop()
        logger.info("Hotkey listener stopped")
