"""Detect macOS Secure Keyboard Entry, which silently deafens global hotkeys.

While any app holds Secure Event Input (a focused password field, Terminal's
"Secure Keyboard Entry", a stuck Chrome login tab), the window server stops
delivering keyDown events to NSEvent global monitors system-wide. The monitor
stays registered and Accessibility stays granted, so without this check the
daemon looks healthy while every hotkey press is dropped.
"""

import ctypes
import logging
import re
import subprocess
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_CARBON_PATH = "/System/Library/Frameworks/Carbon.framework/Carbon"
_PID_RE = re.compile(r'"kCGSSessionSecureInputPID"=(\d+)')

_is_enabled_fn = None


@dataclass(frozen=True)
class SecureInputHolder:
    pid: int
    app: str
    alive: bool = True  # False: holder exited but the flag leaked (macOS bug)


def is_enabled() -> bool:
    """True if some process holds Secure Event Input. Never raises."""
    global _is_enabled_fn
    try:
        if _is_enabled_fn is None:
            fn = ctypes.CDLL(_CARBON_PATH).IsSecureEventInputEnabled
            fn.restype = ctypes.c_bool
            fn.argtypes = []
            _is_enabled_fn = fn
        return bool(_is_enabled_fn())
    except Exception:
        logger.debug("IsSecureEventInputEnabled failed", exc_info=True)
        return False


def _app_name(pid: int) -> str | None:
    """Process name for pid, or None if the process no longer exists."""
    try:
        out = subprocess.run(
            ["ps", "-o", "comm=", "-p", str(pid)],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except Exception:
        return f"pid {pid}"
    if not out:
        return None
    # /Applications/Google Chrome.app/Contents/MacOS/Google Chrome -> Google Chrome
    return out.rsplit("/", 1)[-1]


def find_holder() -> SecureInputHolder | None:
    """Identify the process holding Secure Event Input (slow: shells out to ioreg).

    Call from a background thread only. Returns None if it can't be resolved.
    """
    try:
        out = subprocess.run(
            ["ioreg", "-l", "-w", "0", "-d", "1", "-k", "IOConsoleUsers"],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except Exception:
        logger.debug("ioreg failed", exc_info=True)
        return None
    match = _PID_RE.search(out)
    if match is None:
        return None
    pid = int(match.group(1))
    app = _app_name(pid)
    if app is None:
        return SecureInputHolder(pid=pid, app=f"exited pid {pid}", alive=False)
    return SecureInputHolder(pid=pid, app=app)


class SecureInputWatch:
    """Edge-triggered tracker: reports once when blocking starts and once when it ends."""

    def __init__(self) -> None:
        self._blocked = False

    @property
    def blocked(self) -> bool:
        return self._blocked

    def poll(self) -> str | None:
        """Return "blocked", "cleared", or None when the state is unchanged."""
        now = is_enabled()
        if now == self._blocked:
            return None
        self._blocked = now
        return "blocked" if now else "cleared"
