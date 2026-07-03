"""Manage the macOS LaunchAgent for auto-start and crash supervision."""

import logging
import os
import plistlib
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

LABEL = "com.whisper-daemon"
PLIST_NAME = f"{LABEL}.plist"
LAUNCH_AGENTS_DIR = Path.home() / "Library" / "LaunchAgents"
PLIST_PATH = LAUNCH_AGENTS_DIR / PLIST_NAME
NATIVE_LOG = str(Path.home() / ".config" / "whisper-daemon" / "daemon.native.log")


def _domain() -> str:
    return f"gui/{os.getuid()}"


def is_enabled() -> bool:
    """Check if autostart is currently enabled."""
    return PLIST_PATH.exists()


def is_loaded() -> bool:
    """Check if the launchd job is actually loaded (not just the plist file)."""
    return subprocess.run(
        ["launchctl", "print", f"{_domain()}/{LABEL}"],
        capture_output=True,
    ).returncode == 0


def enable() -> None:
    """Install and (re)load the LaunchAgent.

    Design notes:
    - ``KeepAlive={SuccessfulExit: false}``: launchd restarts the daemon
      after any crash, kill, or watchdog-forced exit (non-zero status),
      but NOT after a user-requested quit (exit 0).
    - ``/bin/sh`` trampoline: launchd records a code-signing requirement
      for ProgramArguments[0]. Pointing it at the venv's python binary
      breaks with OS_REASON_CODESIGNING after every homebrew Python
      upgrade (this silently killed autostart on May 19, 2026). /bin/sh
      never changes; the exec resolves the current python at start time.
    - bootout+bootstrap instead of the deprecated load/unload, so a
      stale registration from an earlier install is fully replaced.
    """
    LAUNCH_AGENTS_DIR.mkdir(parents=True, exist_ok=True)

    plist = {
        "Label": LABEL,
        "ProgramArguments": [
            "/bin/sh", "-c",
            f'exec "{sys.executable}" -m whisper_daemon run',
        ],
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "ThrottleInterval": 10,
        "ProcessType": "Interactive",
        "StandardOutPath": NATIVE_LOG,
        "StandardErrorPath": NATIVE_LOG,
    }

    with open(PLIST_PATH, "wb") as f:
        plistlib.dump(plist, f)

    subprocess.run(
        ["launchctl", "bootout", _domain(), str(PLIST_PATH)],
        capture_output=True,
    )
    result = subprocess.run(
        ["launchctl", "bootstrap", _domain(), str(PLIST_PATH)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        logger.error("launchctl bootstrap failed: %s", result.stderr.strip())
    else:
        logger.info("Autostart enabled and loaded: %s", PLIST_PATH)


def disable() -> None:
    """Unload and remove the LaunchAgent."""
    subprocess.run(
        ["launchctl", "bootout", _domain(), str(PLIST_PATH)],
        capture_output=True,
    )
    if PLIST_PATH.exists():
        PLIST_PATH.unlink()
        logger.info("Autostart disabled: removed %s", PLIST_PATH)
    else:
        logger.info("Autostart already disabled")
