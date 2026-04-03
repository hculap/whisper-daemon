"""Monitor user activity (keyboard, mouse, scroll) to trigger screenshot captures.

Uses NSEvent global monitors for both keyboard and mouse events
(compatible with the AppKit main-thread run loop). Trailing-edge
debounce + cooldown prevents rapid-fire captures during bursts.
"""

import logging
import threading
import time
from collections.abc import Callable

from AppKit import (
    NSEvent,
    NSKeyDownMask,
    NSLeftMouseDownMask,
    NSRightMouseDownMask,
    NSScrollWheelMask,
)

logger = logging.getLogger(__name__)

DEFAULT_DEBOUNCE = 2.0  # seconds after last event before triggering
DEFAULT_COOLDOWN = 5.0  # minimum seconds between captures

MOUSE_EVENT_MASK = NSLeftMouseDownMask | NSRightMouseDownMask | NSScrollWheelMask


class ActivityMonitor:
    """Watches keyboard, mouse clicks, and scroll events.

    Fires a callback after a debounce period of inactivity, with a
    cooldown to prevent rapid-fire triggers.
    """

    def __init__(
        self,
        on_activity: Callable[[], None],
        debounce_delay: float = DEFAULT_DEBOUNCE,
        cooldown: float = DEFAULT_COOLDOWN,
    ) -> None:
        self._on_activity = on_activity
        self._debounce_delay = debounce_delay
        self._cooldown = cooldown

        self._last_capture_time: float = 0.0
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()

        self._keyboard_monitor: object | None = None
        self._mouse_monitor: object | None = None

    def start(self) -> None:
        self._last_capture_time = 0.0

        self._keyboard_monitor = NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(
            NSKeyDownMask,
            self._on_ns_event,
        )

        self._mouse_monitor = NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(
            MOUSE_EVENT_MASK,
            self._on_ns_event,
        )

        logger.info(
            "Activity monitor started (debounce=%.1fs, cooldown=%.1fs)",
            self._debounce_delay,
            self._cooldown,
        )

    def stop(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None

        if self._keyboard_monitor is not None:
            NSEvent.removeMonitor_(self._keyboard_monitor)
            self._keyboard_monitor = None

        if self._mouse_monitor is not None:
            NSEvent.removeMonitor_(self._mouse_monitor)
            self._mouse_monitor = None

        logger.info("Activity monitor stopped")

    def _on_ns_event(self, _event: object) -> None:
        self._on_event()

    def _on_event(self) -> None:
        """Reset debounce timer on any tracked event."""
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()

            self._timer = threading.Timer(self._debounce_delay, self._fire)
            self._timer.daemon = True
            self._timer.start()

    def _fire(self) -> None:
        """Trigger capture if cooldown has elapsed."""
        now = time.monotonic()

        with self._lock:
            self._timer = None
            elapsed_since_capture = now - self._last_capture_time
            if elapsed_since_capture < self._cooldown:
                logger.debug(
                    "Activity trigger skipped — cooldown (%.1fs remaining)",
                    self._cooldown - elapsed_since_capture,
                )
                return
            self._last_capture_time = now

        logger.debug("Activity trigger — capturing screenshot")
        try:
            self._on_activity()
        except Exception:
            logger.exception("Activity-triggered capture failed")
