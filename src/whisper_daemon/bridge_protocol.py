"""Pure protocol helpers for the browser bridge.

This module is deliberately dependency-light: it imports only the standard
library and :mod:`whisper_daemon.config`. It MUST NOT import mlx/torch/pyobjc
or anything heavy, so it can be unit-tested in isolation and reused from any
thread without side effects.

All functions are pure. Settings mutations return a NEW ``Settings`` via
``dataclasses.replace`` — the input is never modified.
"""

from __future__ import annotations

import dataclasses
import json

from .config import VALID_FORMATS, Settings

ALLOWED_ORIGIN_PREFIX = "chrome-extension://"

# Valid enumerations reused during patch validation.
_VALID_SCREENSHOT_DISPLAYS = {"all", "primary"}
_VALID_DIARIZE_MODES = {"batch", "realtime", "hybrid"}

# Settings fields that must be plain booleans.
_BOOL_FIELDS = {
    "capture_mic",
    "capture_tab",
    "capture_screenshots",
    "live_captions",
    "diarize",
}
# Settings fields that must be plain strings (free-form).
_STR_FIELDS = {"recording_device", "recording_dir"}

# Characters that could break out of a TOML basic string when the value is
# later persisted to config.toml. A patch carrying any of these for a free-form
# string field is rejected (the old value is kept) so set_settings cannot inject
# or corrupt config keys.
_UNSAFE_STR_CHARS = ('"', "\\", "\n", "\r", "\x00")


def _is_safe_str(value: str) -> bool:
    """Reject free-form strings that could corrupt/inject config on save."""
    return not any(ch in value for ch in _UNSAFE_STR_CHARS)


def is_allowed_origin(origin: str | None) -> bool:
    """Return True only for chrome-extension:// origins.

    Websites (http/https) and a missing Origin header are rejected so the
    daemon's WebSocket surface cannot be driven by an arbitrary web page.
    """
    if origin is None:
        return False
    return origin.startswith(ALLOWED_ORIGIN_PREFIX)


def settings_to_dict(s: Settings) -> dict:
    """Serialize a Settings into the exact extension snapshot shape."""
    return {
        "recording_device": s.recording_device,
        "capture_mic": s.capture_mic,
        "capture_tab": s.capture_tab,
        "capture_screenshots": s.capture_screenshots,
        "screenshot_displays": s.screenshot_displays,
        "live_captions": s.live_captions,
        "diarize": s.diarize,
        "diarize_mode": s.diarize_mode,
        "recording_dir": s.recording_dir,
        # copy so callers can't mutate the Settings' underlying list
        "recording_formats": list(s.recording_formats),
    }


def build_settings_message(s: Settings, devices: list[str]) -> str:
    """Build the JSON 'settings' message (snapshot + available devices)."""
    return json.dumps({
        "type": "settings",
        "settings": settings_to_dict(s),
        "devices": list(devices),
    })


def build_status_message(state: str, elapsed: float, chunks: int) -> str:
    """Build the JSON 'status' message pushed to the extension."""
    return json.dumps({
        "type": "status",
        "state": state,
        "elapsed": elapsed,
        "chunks": chunks,
    })


def parse_client_message(raw: str) -> dict | None:
    """Parse a text frame into a dict, or None on error / non-object JSON."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def _clean_formats(value: object, current: list[str]) -> list[str]:
    """Keep only VALID_FORMATS from value; fall back to current if empty."""
    if not isinstance(value, list):
        return current
    cleaned = [f for f in value if isinstance(f, str) and f in VALID_FORMATS]
    return cleaned if cleaned else current


def apply_settings_patch(s: Settings, patch: dict) -> Settings:
    """Return a NEW Settings with validated fields from patch applied.

    Immutable: never mutates ``s``. Unknown keys are ignored, invalid values
    are skipped (the old value is kept). Booleans are coerced strictly
    (``True``/``False`` only — ``1``/``"true"`` are rejected). Enumerated
    strings are validated against their allowed sets.
    """
    if not isinstance(patch, dict):
        return s

    updates: dict[str, object] = {}

    for key, value in patch.items():
        if key in _BOOL_FIELDS:
            if isinstance(value, bool):
                updates[key] = value
        elif key in _STR_FIELDS:
            # Reject values that could inject/corrupt TOML on persist — keep old.
            if isinstance(value, str) and _is_safe_str(value):
                updates[key] = value
        elif key == "screenshot_displays":
            if isinstance(value, str) and value in _VALID_SCREENSHOT_DISPLAYS:
                updates[key] = value
        elif key == "diarize_mode":
            if isinstance(value, str) and value in _VALID_DIARIZE_MODES:
                updates[key] = value
        elif key == "recording_formats":
            updates[key] = _clean_formats(value, s.recording_formats)
        # unknown keys are ignored

    if not updates:
        return s
    return dataclasses.replace(s, **updates)
