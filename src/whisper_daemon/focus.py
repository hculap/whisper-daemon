"""Capture and compare the actual macOS app, window and focused input element."""
from dataclasses import dataclass

@dataclass(frozen=True)
class Target:
    pid: int
    window: object
    element: object


def capture_target():
    try:
        from AppKit import NSWorkspace
        from ApplicationServices import AXUIElementCreateApplication, AXUIElementCopyAttributeValue
        app = NSWorkspace.sharedWorkspace().frontmostApplication()
        if app is None: return None
        pid = app.processIdentifier()
        element = AXUIElementCreateApplication(pid)
        error, window = AXUIElementCopyAttributeValue(element, 'AXFocusedWindow', None)
        if error: return None
        error, field = AXUIElementCopyAttributeValue(element, 'AXFocusedUIElement', None)
        if error or field is None: return None
        error, subrole = AXUIElementCopyAttributeValue(field, 'AXSubrole', None)
        if subrole == 'AXSecureTextField': return None
        return Target(pid, window, field)
    except Exception:
        return None


def target_matches(target):
    return target is not None and target == capture_target()
