"""Paste text into the active application via clipboard + Cmd+V."""

import logging
import time

from AppKit import NSPasteboard, NSPasteboardTypeString
from Quartz import (
    CGEventCreateKeyboardEvent,
    CGEventPost,
    CGEventSetFlags,
    kCGEventFlagMaskCommand,
    kCGHIDEventTap,
)

logger = logging.getLogger(__name__)

V_KEYCODE = 9  # macOS virtual keycode for 'v'
PASTE_SETTLE_DELAY = 0.2  # seconds to wait for paste to complete


def paste_text(text: str) -> None:
    """Paste text into the currently focused text field.

    Saves the current clipboard, sets the transcription, simulates Cmd+V,
    then restores the original clipboard content.
    """
    pb = NSPasteboard.generalPasteboard()

    old_content = pb.stringForType_(NSPasteboardTypeString)

    pb.clearContents()
    pb.setString_forType_(text, NSPasteboardTypeString)

    _simulate_cmd_v()

    time.sleep(PASTE_SETTLE_DELAY)

    pb.clearContents()
    if old_content is not None:
        pb.setString_forType_(old_content, NSPasteboardTypeString)

    logger.info("Pasted %d chars", len(text))


def _simulate_cmd_v() -> None:
    """Simulate Cmd+V keypress via CGEvent."""
    event_down = CGEventCreateKeyboardEvent(None, V_KEYCODE, True)
    CGEventSetFlags(event_down, kCGEventFlagMaskCommand)
    CGEventPost(kCGHIDEventTap, event_down)

    event_up = CGEventCreateKeyboardEvent(None, V_KEYCODE, False)
    CGEventSetFlags(event_up, kCGEventFlagMaskCommand)
    CGEventPost(kCGHIDEventTap, event_up)
