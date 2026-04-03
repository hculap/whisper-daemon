"""Global hotkey listener using NSEvent (replaces pynput GlobalHotKeys).

NSEvent global monitors are compatible with the AppKit main-thread run loop
and avoid the pynput/Python 3.14 signature mismatch that silently kills
the listener thread.
"""

import logging
import queue

from AppKit import NSEvent, NSKeyDownMask, NSCommandKeyMask, NSShiftKeyMask

from whisper_daemon.events import Event, EventType

logger = logging.getLogger(__name__)

RECORD_COMBO = (NSCommandKeyMask | NSShiftKeyMask, 49)       # Cmd+Shift+Space
PASTE_LAST_COMBO = (NSCommandKeyMask | NSShiftKeyMask, 9)    # Cmd+Shift+V

HOTKEY_DESCRIPTIONS = {
    RECORD_COMBO: "Cmd+Shift+Space (record)",
    PASTE_LAST_COMBO: "Cmd+Shift+V (paste last)",
}


class HotkeyListener:
    """Listens for global hotkeys via NSEvent global key-down monitor."""

    def __init__(self, event_queue: queue.Queue[Event]) -> None:
        self._queue = event_queue
        self._monitor: object | None = None

    def _handle_key(self, ns_event: object) -> None:
        flags = ns_event.modifierFlags()
        keycode = ns_event.keyCode()

        modifier_mask = NSCommandKeyMask | NSShiftKeyMask
        if (flags & modifier_mask) != modifier_mask:
            return

        if keycode == RECORD_COMBO[1]:
            logger.info("Hotkey: record toggle")
            self._queue.put(Event(EventType.RECORD_TOGGLE))
        elif keycode == PASTE_LAST_COMBO[1]:
            logger.info("Hotkey: paste last transcription")
            self._queue.put(Event(EventType.PASTE_LAST))

    def start(self) -> None:
        self._monitor = NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(
            NSKeyDownMask,
            self._handle_key,
        )
        for combo, desc in HOTKEY_DESCRIPTIONS.items():
            logger.info("Hotkey registered: %s", desc)

    def stop(self) -> None:
        if self._monitor is not None:
            NSEvent.removeMonitor_(self._monitor)
            self._monitor = None
        logger.info("Hotkey listener stopped")
