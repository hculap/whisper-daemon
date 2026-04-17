"""Manage macOS LaunchAgent for auto-start at login."""

import logging
import plistlib
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

PLIST_NAME = "com.whisper-daemon.plist"
LAUNCH_AGENTS_DIR = Path.home() / "Library" / "LaunchAgents"
PLIST_PATH = LAUNCH_AGENTS_DIR / PLIST_NAME


def is_enabled() -> bool:
    """Check if autostart is currently enabled."""
    return PLIST_PATH.exists()


def enable() -> None:
    """Create LaunchAgent plist to start whisper-daemon at login."""
    LAUNCH_AGENTS_DIR.mkdir(parents=True, exist_ok=True)

    plist = {
        "Label": "com.whisper-daemon",
        "ProgramArguments": [
            sys.executable, "-m", "whisper_daemon", "run",
        ],
        "RunAtLoad": True,
        "KeepAlive": False,
        "StandardOutPath": str(Path.home() / ".config" / "whisper-daemon" / "daemon.native.log"),
        "StandardErrorPath": str(Path.home() / ".config" / "whisper-daemon" / "daemon.native.log"),
    }

    with open(PLIST_PATH, "wb") as f:
        plistlib.dump(plist, f)

    logger.info("Autostart enabled: %s", PLIST_PATH)


def disable() -> None:
    """Remove LaunchAgent plist."""
    if PLIST_PATH.exists():
        PLIST_PATH.unlink()
        logger.info("Autostart disabled: removed %s", PLIST_PATH)
    else:
        logger.info("Autostart already disabled")
