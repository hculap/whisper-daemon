"""Global hotkey listener: Carbon hot key for record, NSEvent monitor for the rest.

Record (Cmd+Shift+Space) is a Carbon RegisterEventHotKey because it is the
only mechanism that survives Secure Keyboard Entry (password fields, iTerm2
auto secure input) — the NSEvent global monitor goes silently deaf then.
Paste-last and copy-and-speak stay on the passive NSEvent monitor: Carbon
consumes its combos, which would break Cmd+Shift+V / Cmd+Shift+C in other
apps. If Carbon registration fails, record falls back to the monitor too.
"""

import logging
import queue
import threading
import time

import objc
from AppKit import (
    NSEvent,
    NSKeyDownMask,
    NSCommandKeyMask,
    NSShiftKeyMask,
    NSObject,
    NSWorkspace,
)

from whisper_daemon import secure_input
from whisper_daemon.carbon_hotkey import CMD_KEY, SHIFT_KEY, CarbonHotkeys
from whisper_daemon.events import Event, EventType
from whisper_daemon.notify import notify

logger = logging.getLogger(__name__)

RECORD_COMBO = (NSCommandKeyMask | NSShiftKeyMask, 49)       # Cmd+Shift+Space
PASTE_LAST_COMBO = (NSCommandKeyMask | NSShiftKeyMask, 9)    # Cmd+Shift+V
SPEAK_COMBO = (NSCommandKeyMask | NSShiftKeyMask, 8)         # Cmd+Shift+C

HOTKEY_DESCRIPTIONS = {
    RECORD_COMBO: "Cmd+Shift+Space (record)",
    PASTE_LAST_COMBO: "Cmd+Shift+V (paste last)",
    SPEAK_COMBO: "Cmd+Shift+C (copy & speak)",
}

HEARTBEAT_INTERVAL_S = 600.0  # 10 minutes
REINSTALL_EVERY_BEATS = 36  # reinstall the monitor every 36 beats = 6h
SECURE_INPUT_POLL_S = 15.0  # how fast a Secure-Input block gets reported


class _WakeObserver(NSObject):
    """Relays NSWorkspaceDidWakeNotification to a Python callable.

    NSNotificationCenter requires an NSObject with an ObjC selector; the
    HotkeyListener itself is a plain Python object, so we use this thin
    wrapper and keep a strong reference on the listener.
    """

    def initWithCallback_(self, callback):
        self = objc.super(_WakeObserver, self).init()
        if self is None:
            return None
        self._callback = callback
        return self

    @objc.typedSelector(b"v@:@")
    def onWake_(self, _notification):
        try:
            self._callback()
        except Exception:
            logger.exception("Wake callback failed")


class HotkeyListener:
    """Listens for global hotkeys via NSEvent global key-down monitor.

    Re-registers the monitor on macOS wake (NSWorkspaceDidWakeNotification)
    since the global monitor can silently stop delivering events after long
    sleep/wake cycles. Also emits a periodic heartbeat so the log makes it
    obvious when the listener is alive vs wedged.
    """

    def __init__(self, event_queue: queue.Queue[Event]) -> None:
        self._queue = event_queue
        self._monitor: object | None = None
        self._wake_observer: _WakeObserver | None = None
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._last_event_at: float = 0.0
        self._secure_watch = secure_input.SecureInputWatch()
        self._secure_holder_pid: int | None = None
        self._carbon = CarbonHotkeys()
        self._record_via_carbon = False

    def _handle_key(self, ns_event: object) -> None:
        flags = ns_event.modifierFlags()
        keycode = ns_event.keyCode()

        modifier_mask = NSCommandKeyMask | NSShiftKeyMask
        if (flags & modifier_mask) != modifier_mask:
            return

        self._last_event_at = time.monotonic()

        if keycode == RECORD_COMBO[1]:
            if not self._record_via_carbon:
                self._on_record()
        elif keycode == PASTE_LAST_COMBO[1]:
            logger.info("Hotkey: paste last transcription")
            self._queue.put(Event(EventType.PASTE_LAST))
        elif keycode == SPEAK_COMBO[1]:
            logger.info("Hotkey: copy & speak")
            self._queue.put(Event(EventType.SPEAK_CLIPBOARD))

    def _on_record(self) -> None:
        self._last_event_at = time.monotonic()
        logger.info("Hotkey: record toggle")
        self._queue.put(Event(EventType.RECORD_TOGGLE))

    def _register_carbon_record(self) -> None:
        try:
            self._carbon.register(
                RECORD_COMBO[1], CMD_KEY | SHIFT_KEY, self._on_record,
            )
            self._record_via_carbon = True
        except OSError:
            logger.exception(
                "Carbon record hotkey unavailable — falling back to NSEvent "
                "monitor (record will not work under Secure Keyboard Entry)"
            )
            self._record_via_carbon = False

    def _install_monitor(self) -> None:
        if self._monitor is not None:
            NSEvent.removeMonitor_(self._monitor)
            self._monitor = None
        self._monitor = NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(
            NSKeyDownMask,
            self._handle_key,
        )

    def _on_wake(self) -> None:
        # Rebuild the global monitor — after long sleeps the existing one can
        # stop receiving events even though it's still registered. Cheap to
        # redo; worst case it's a no-op refresh.
        logger.info("Hotkey: system wake — reinstalling global monitor")
        self._install_monitor()

    def _check_secure_input(self) -> None:
        """Log Secure Input changes; notify only if it blocks the record hotkey."""
        change = self._secure_watch.poll()
        if change == "cleared":
            self._secure_holder_pid = None
            logger.info("Secure Keyboard Entry released — all hotkeys active")
            return
        if not self._secure_watch.blocked:
            return
        # Re-resolve while blocked: the holder can hand over (e.g. Chrome
        # quits, iTerm2 takes it) without the flag ever dropping.
        holder = secure_input.find_holder()
        holder_pid = holder.pid if holder else None
        if change != "blocked" and holder_pid == self._secure_holder_pid:
            return
        self._secure_holder_pid = holder_pid

        if holder is None:
            who = "another app"
        elif holder.alive:
            who = f"{holder.app} (pid {holder.pid})"
        else:
            who = f"exited pid {holder.pid} — stuck flag, lock/unlock screen to reset"

        if self._record_via_carbon:
            logger.info(
                "Secure Keyboard Entry held by %s — Cmd+Shift+V/C paused, "
                "record hotkey unaffected", who,
            )
            return

        logger.warning("Hotkeys BLOCKED — Secure Keyboard Entry held by %s", who)
        from PyObjCTools import AppHelper

        AppHelper.callAfter(
            notify, "whisper-daemon", "Hotkeys blocked",
            f"Secure Keyboard Entry held by {who}.",
        )

    def _heartbeat_loop(self) -> None:
        beats = 0
        ticks_per_beat = max(1, round(HEARTBEAT_INTERVAL_S / SECURE_INPUT_POLL_S))
        ticks = 0
        self._check_secure_input()
        while not self._heartbeat_stop.wait(SECURE_INPUT_POLL_S):
            self._check_secure_input()
            ticks += 1
            if ticks % ticks_per_beat:
                continue
            beats += 1
            since = time.monotonic() - self._last_event_at if self._last_event_at else -1
            logger.info(
                "Hotkey heartbeat — monitor=%s, secure_input=%s, last_event=%s",
                "alive" if self._monitor is not None else "none",
                "BLOCKING" if self._secure_watch.blocked else "off",
                f"{since:.0f}s ago" if since >= 0 else "never",
            )
            # Periodic reinstall: the monitor can die without any wake
            # notification (long uptime, TCC permission flaps). Rebuilding
            # it is instant and idempotent, so refresh it every 6h as
            # insurance. Must run on the main thread.
            if beats % REINSTALL_EVERY_BEATS == 0:
                from PyObjCTools import AppHelper

                logger.info("Hotkey: periodic monitor reinstall")
                AppHelper.callAfter(self._install_monitor)

    def start(self) -> None:
        self._register_carbon_record()
        self._install_monitor()

        self._wake_observer = _WakeObserver.alloc().initWithCallback_(self._on_wake)
        center = NSWorkspace.sharedWorkspace().notificationCenter()
        # Reinstall on every "the session came back" signal — plain wake,
        # display wake, and fast-user-switch reactivation each have their
        # own notification and any of them can leave the monitor deaf.
        for note in (
            "NSWorkspaceDidWakeNotification",
            "NSWorkspaceScreensDidWakeNotification",
            "NSWorkspaceSessionDidBecomeActiveNotification",
        ):
            center.addObserver_selector_name_object_(
                self._wake_observer, "onWake:", note, None,
            )

        self._heartbeat_stop.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name="hotkey-heartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()

        for _, desc in HOTKEY_DESCRIPTIONS.items():
            logger.info("Hotkey registered: %s", desc)

    def stop(self) -> None:
        self._carbon.unregister_all()
        self._record_via_carbon = False
        if self._monitor is not None:
            NSEvent.removeMonitor_(self._monitor)
            self._monitor = None
        if self._wake_observer is not None:
            NSWorkspace.sharedWorkspace().notificationCenter().removeObserver_(
                self._wake_observer
            )
            self._wake_observer = None
        self._heartbeat_stop.set()
        logger.info("Hotkey listener stopped")
