"""Paste text into the active application via clipboard + Cmd+V."""

import logging
import time

from AppKit import NSPasteboard, NSPasteboardItem, NSPasteboardTypeString
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

    saved = []
    for original in pb.pasteboardItems() or []:
        item = NSPasteboardItem.alloc().init()
        for content_type in original.types():
            data = original.dataForType_(content_type)
            if data is not None:
                item.setData_forType_(data, content_type)
        saved.append(item)
    pb.clearContents()
    pb.setString_forType_(text, NSPasteboardTypeString)
    owned_version = pb.changeCount()
    try:
        _simulate_cmd_v()
        time.sleep(PASTE_SETTLE_DELAY)
    finally:
        # A copy performed by the user during paste belongs to the user.
        if pb.changeCount() == owned_version:
            pb.clearContents()
            if saved:
                pb.writeObjects_(saved)

    logger.info("Pasted %d chars", len(text))


def _simulate_cmd_v() -> None:
    """Simulate Cmd+V keypress via CGEvent."""
    event_down = CGEventCreateKeyboardEvent(None, V_KEYCODE, True)
    CGEventSetFlags(event_down, kCGEventFlagMaskCommand)
    CGEventPost(kCGHIDEventTap, event_down)

    event_up = CGEventCreateKeyboardEvent(None, V_KEYCODE, False)
    CGEventSetFlags(event_up, kCGEventFlagMaskCommand)
    CGEventPost(kCGHIDEventTap, event_up)
