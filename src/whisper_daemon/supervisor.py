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

from collections.abc import Callable, Iterator
from contextlib import contextmanager
import logging
import os
import threading
import time

from whisper_daemon.memory_guard import DEFAULT_LIMIT_BYTES, MemoryGuard

logger = logging.getLogger(__name__)

RESTART_EXIT_CODE = 86
CHECK_INTERVAL_SEC = 15.0
MEMORY_CHECK_INTERVAL_SEC = 60.0

# name -> max allowed seconds between beats
_LIMITS = {
    "event-loop": 120.0,
    "main-thread": 120.0,
}

_beats: dict[str, float] = {}
_beats_lock = threading.Lock()
_watchdog_thread: threading.Thread | None = None
_pre_exit_hooks: list[Callable[[], None]] = []
_suspended = False
_memory_guard: MemoryGuard | None = None
# Held only to count transitions and across the restart commit itself, never
# across a transition's I/O, so transitions on different threads (main, event
# loop, browser bridge) cannot stall one another.
_activity_gate = threading.RLock()
_active_transitions = 0


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


def add_pre_exit_hook(hook: Callable[[], None]) -> None:
    """Register a callable to run (best-effort) before a forced restart."""
    _pre_exit_hooks.append(hook)


@contextmanager
def activity_transition() -> Iterator[None]:
    """Mark new user work as starting so a memory restart cannot commit.

    The transition body runs without the gate: it only blocks for the instant
    a restart commits, after which the process exits anyway.
    """
    global _active_transitions
    with _activity_gate:
        _active_transitions += 1
    try:
        yield
    finally:
        with _activity_gate:
            _active_transitions -= 1


def _work_active(is_idle: Callable[[], bool], label: str) -> bool:
    """Final busy check; the caller holds the activity gate."""
    if _active_transitions:
        logger.info("%s deferred because new work is starting", label)
        return True
    try:
        if is_idle():
            return False
    except Exception:
        logger.exception("%s safety check failed", label)
        return True
    logger.info("%s deferred because new work started", label)
    return True


@contextmanager
def _idle_gate(is_idle: Callable[[], bool], label: str) -> Iterator[bool]:
    """Yield whether the daemon is idle while holding the gate.

    Never blocks: the only caller is the watchdog thread, which must keep
    enforcing heartbeats even while a transition is wedged.
    """
    if not _activity_gate.acquire(blocking=False):
        logger.info("%s deferred because protected work is active", label)
        yield False
        return
    try:
        yield not _work_active(is_idle, label)
    finally:
        _activity_gate.release()


def request_restart_if_safe(reason: str, is_idle: Callable[[], bool]) -> bool:
    """Atomically recheck idle and commit a memory-pressure restart.

    The process exits while still holding the gate, so once idle has been
    observed here no new protected work can begin before it is gone.
    """
    with _idle_gate(is_idle, "Memory restart") as idle:
        if idle:
            request_restart(reason)
        return idle


def reclaim_if_safe(reclaim: Callable[[], None], is_idle: Callable[[], bool]) -> bool:
    """Reclaim caches only at an idle transition, without blocking watchdogs."""
    with _idle_gate(is_idle, "Memory reclamation") as idle:
        pass
    # Outside the gate: gc and MPS synchronization can take seconds. Work that
    # starts meanwhile is serialized against the cleanup by MLX_LOCK and the
    # diarizer lock, and at worst reloads a model.
    if idle:
        reclaim()
    return idle


def configure_memory_guard(
    is_idle: Callable[[], bool],
    reclaim: Callable[[], None],
    *,
    limit_bytes: int = DEFAULT_LIMIT_BYTES,
) -> None:
    """Enable physical-footprint enforcement for this process."""
    global _memory_guard
    _memory_guard = MemoryGuard(
        is_idle=is_idle,
        reclaim=lambda: reclaim_if_safe(reclaim, is_idle),
        restart=lambda reason: request_restart_if_safe(reason, is_idle),
        limit_bytes=limit_bytes,
    )
    logger.info("Memory guard enabled (limit %.1f GiB)", limit_bytes / 1024**3)


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
    last_memory_check = 0.0
    while True:
        time.sleep(CHECK_INTERVAL_SEC)
        if _suspended:
            continue
        now = time.monotonic()
        if _memory_guard is not None and now - last_memory_check >= MEMORY_CHECK_INTERVAL_SEC:
            last_memory_check = now
            _memory_guard.check()
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
