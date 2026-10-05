"""macOS user notifications (best-effort, never raises)."""

import logging

logger = logging.getLogger(__name__)


def notify(title: str, subtitle: str, message: str) -> None:
    """Post a macOS notification via NSUserNotificationCenter (best-effort)."""
    try:
        from Foundation import (
            NSUserNotification,
            NSUserNotificationCenter,
        )

        notification = NSUserNotification.alloc().init()
        notification.setTitle_(title)
        notification.setSubtitle_(subtitle)
        notification.setInformativeText_(message)
        NSUserNotificationCenter.defaultUserNotificationCenter().deliverNotification_(
            notification
        )
    except Exception:
        logger.warning("Notification failed: %s — %s — %s", title, subtitle, message)
