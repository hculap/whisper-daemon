"""macOS Accessibility permission handling for global hotkeys.

The global NSEvent key-down monitor only receives events when the process
is trusted for Accessibility. That trust is bound to the specific process
identity (interpreter binary + how it was launched), NOT inherited from a
parent — so a daemon started by launchd needs its OWN grant even if the
same binary works when launched from a trusted Terminal. Switching Python
builds or launch method silently breaks hotkeys with no error, only a
dead monitor. This module detects that state and asks the user to grant
access via the standard system dialog.

Uses the pyobjc ApplicationServices bridge (safe, memory-managed) — an
earlier hand-rolled ctypes CFDictionary crashed the process, which under
launchd KeepAlive became a restart loop, so everything here is defensive
and never raises.
"""

import logging

logger = logging.getLogger(__name__)


def is_trusted() -> bool:
    """True if this process is trusted for Accessibility (no dialog)."""
    try:
        from ApplicationServices import AXIsProcessTrusted
        return bool(AXIsProcessTrusted())
    except Exception:
        logger.debug("AXIsProcessTrusted check failed", exc_info=True)
        # Assume trusted so a probe failure never blocks startup.
        return True


def request_access() -> bool:
    """Check trust and, if missing, show the system Accessibility prompt.

    The dialog adds this process to System Settings > Privacy & Security >
    Accessibility (unchecked); the grant only takes effect after the user
    enables it and the daemon restarts. Returns the current trust state.
    Never raises.
    """
    try:
        from ApplicationServices import (
            AXIsProcessTrustedWithOptions,
            kAXTrustedCheckOptionPrompt,
        )
        options = {kAXTrustedCheckOptionPrompt: True}
        return bool(AXIsProcessTrustedWithOptions(options))
    except Exception:
        logger.debug("Accessibility prompt failed", exc_info=True)
        return is_trusted()


def check_and_warn() -> bool:
    """Log the Accessibility state; prompt + warn loudly if missing.

    Returns the trust state. Never raises.
    """
    if request_access():
        logger.info("Accessibility permission: granted (global hotkeys active)")
        return True
    logger.warning(
        "Accessibility permission MISSING — global hotkeys will NOT work. "
        "A system dialog was requested: open System Settings > Privacy & "
        "Security > Accessibility, enable 'Python' (whisper-daemon), then "
        "run 'whisper-daemon restart'."
    )
    return False
