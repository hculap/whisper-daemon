"""Memory-pressure backstop for the long-running macOS daemon.

The models use Apple unified memory.  Ordinary RSS misses a large part of
Metal/IOAccelerator allocations, so the guard deliberately reads the same
``phys_footprint`` metric that macOS uses for memory pressure accounting.
"""

from collections.abc import Callable
import ctypes
import logging
import os
import sys
import time

logger = logging.getLogger(__name__)

GIB = 1024**3
DEFAULT_LIMIT_BYTES = 4 * GIB
DEFAULT_STARTUP_GRACE_SEC = 300.0


class _RUsageInfoV0(ctypes.Structure):
    _fields_ = [
        ("ri_uuid", ctypes.c_uint8 * 16),
        ("ri_user_time", ctypes.c_uint64),
        ("ri_system_time", ctypes.c_uint64),
        ("ri_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_interrupt_wkups", ctypes.c_uint64),
        ("ri_pageins", ctypes.c_uint64),
        ("ri_wired_size", ctypes.c_uint64),
        ("ri_resident_size", ctypes.c_uint64),
        ("ri_phys_footprint", ctypes.c_uint64),
        ("ri_proc_start_abstime", ctypes.c_uint64),
        ("ri_proc_exit_abstime", ctypes.c_uint64),
    ]


_libproc = None


def physical_footprint_bytes(pid: int | None = None) -> int | None:
    """Return macOS physical footprint, including compressed/Metal memory."""
    if sys.platform != "darwin":
        return None

    global _libproc
    try:
        if _libproc is None:
            _libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
            _libproc.proc_pid_rusage.argtypes = [
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_void_p,
            ]
            _libproc.proc_pid_rusage.restype = ctypes.c_int

        usage = _RUsageInfoV0()
        # RUSAGE_INFO_V0 is enough: it already contains ri_phys_footprint.
        result = _libproc.proc_pid_rusage(
            os.getpid() if pid is None else pid,
            0,
            ctypes.byref(usage),
        )
        if result != 0:
            return None
        return int(usage.ri_phys_footprint)
    except Exception:
        logger.exception("Could not read process physical footprint")
        return None


class MemoryGuard:
    """Reclaim once, then restart only if a high footprint persists at idle."""

    def __init__(
        self,
        *,
        measure: Callable[[], int | None] = physical_footprint_bytes,
        is_idle: Callable[[], bool],
        reclaim: Callable[[], bool | None],
        restart: Callable[[str], bool | None],
        limit_bytes: int = DEFAULT_LIMIT_BYTES,
        startup_grace_sec: float = DEFAULT_STARTUP_GRACE_SEC,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._measure = measure
        self._is_idle = is_idle
        self._reclaim = reclaim
        self._restart = restart
        self._limit_bytes = limit_bytes
        self._startup_grace_sec = startup_grace_sec
        self._clock = clock
        self._started_at = clock()
        self._reclaimed = False
        self._restart_requested = False
        self._over_limit_logged = False

    def check(self) -> None:
        """Take one sample and defer action while protected work is active."""
        if self._restart_requested:
            return
        if self._clock() - self._started_at < self._startup_grace_sec:
            return

        try:
            footprint = self._measure()
        except Exception:
            logger.exception("Memory footprint measurement failed")
            return
        if footprint is None:
            return

        if footprint <= self._limit_bytes:
            if self._over_limit_logged:
                logger.info(
                    "Memory footprint recovered to %.2f GiB",
                    footprint / GIB,
                )
            self._reclaimed = False
            self._over_limit_logged = False
            return

        if not self._over_limit_logged:
            logger.warning(
                "Physical footprint %.2f GiB exceeds %.2f GiB; waiting for idle",
                footprint / GIB,
                self._limit_bytes / GIB,
            )
            self._over_limit_logged = True

        try:
            idle = self._is_idle()
        except Exception:
            logger.exception("Memory guard idle probe failed; treating daemon as busy")
            return
        if not idle:
            return

        if not self._reclaimed:
            try:
                accepted = self._reclaim()
            except Exception:
                logger.exception("Memory reclamation failed; will retry while idle")
                return
            if accepted is False:
                return
            self._reclaimed = True
            logger.warning("Released inference caches after high memory footprint")
            return

        reason = (
            f"physical footprint remains {footprint / GIB:.1f} GiB "
            f"after cache cleanup (limit {self._limit_bytes / GIB:.1f} GiB)"
        )
        # A coordinator may perform one last idle check under a transition
        # lock.  ``False`` means new work won that race, so keep monitoring and
        # try again after that work finishes.
        accepted = self._restart(reason)
        if accepted is not False:
            self._restart_requested = True
