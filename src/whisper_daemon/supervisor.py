"""In-process liveness supervision.

Threads that must stay responsive call ``beat(name)`` regularly; the
watchdog thread checks the age of every beat against its limit and, when
one goes stale, force-exits the process with ``RESTART_EXIT_CODE``. The
launchd job runs with ``KeepAlive={SuccessfulExit: false}``, so any
non-zero exit — a wedged main thread, a dead event loop, a hung MLX
inference — turns into an automatic restart instead of a silent outage.

``time.monotonic()`` pauses during system sleep on macOS, so long sleeps
do not produce false positives.
"""

import logging
import os
import threading
import time

logger = logging.getLogger(__name__)

RESTART_EXIT_CODE = 86
CHECK_INTERVAL_SEC = 15.0

# name -> max allowed seconds between beats
_LIMITS = {
    "event-loop": 120.0,
    "main-thread": 120.0,
}

_beats: dict[str, float] = {}
_beats_lock = threading.Lock()
_watchdog_thread: threading.Thread | None = None
_pre_exit_hooks: list = []
_suspended = False


def beat(name: str) -> None:
    """Record a liveness heartbeat for the named component."""
    with _beats_lock:
        _beats[name] = time.monotonic()


def suspend(reason: str) -> None:
    """Stop liveness enforcement — the process is shutting down on purpose.

    During a user-requested quit the poll timer stops and the main thread
    may legitimately block on a meeting save for minutes; without this,
    the watchdog would exit non-zero and launchd would resurrect a daemon
    the user just stopped. The quit path's own exit watchdog (exit code 0)
    owns termination from here.
    """
    global _suspended
    _suspended = True
    logger.info("Liveness supervision suspended: %s", reason)


def add_pre_exit_hook(hook) -> None:
    """Register a callable to run (best-effort) before a forced restart."""
    _pre_exit_hooks.append(hook)


def request_restart(reason: str) -> None:
    """Log, run pre-exit hooks, and exit so launchd restarts the daemon.

    If supervision is suspended (user-requested shutdown in progress),
    exit 0 instead — the user asked for a stop, not a restart.
    """
    exit_code = 0 if _suspended else RESTART_EXIT_CODE
    logger.critical("Forced exit (%d): %s", exit_code, reason)
    for hook in _pre_exit_hooks:
        try:
            hook()
        except Exception:
            logger.exception("Pre-exit hook failed")
    logging.shutdown()
    os._exit(exit_code)


def start() -> None:
    """Start the liveness watchdog thread (idempotent)."""
    global _watchdog_thread
    if _watchdog_thread is not None and _watchdog_thread.is_alive():
        return
    _watchdog_thread = threading.Thread(
        target=_watchdog_loop, name="liveness-watchdog", daemon=True
    )
    _watchdog_thread.start()
    logger.info("Liveness watchdog started (limits: %s)", _LIMITS)


def _watchdog_loop() -> None:
    while True:
        time.sleep(CHECK_INTERVAL_SEC)
        if _suspended:
            continue
        now = time.monotonic()
        with _beats_lock:
            snapshot = dict(_beats)
        for name, last in snapshot.items():
            limit = _LIMITS.get(name)
            if limit is None:
                continue
            age = now - last
            if age > limit:
                request_restart(
                    f"'{name}' unresponsive for {age:.0f}s (limit {limit:.0f}s)"
                )
