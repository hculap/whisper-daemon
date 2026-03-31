"""Menu bar status icon using rumps."""

import logging
import threading

import rumps

from whisper_daemon.daemon import State

logger = logging.getLogger(__name__)

ICONS = {
    State.IDLE: "🎙",
    State.RECORDING: "🔴",
    State.TRANSCRIBING: "⏳",
}

TITLES = {
    State.IDLE: "Ready",
    State.RECORDING: "Recording...",
    State.TRANSCRIBING: "Transcribing...",
}


class MenuBarApp(rumps.App):
    """Menu bar icon showing daemon state."""

    def __init__(self, daemon: object, hotkey_listener: object) -> None:
        super().__init__(ICONS[State.IDLE], quit_button=None)
        self._daemon = daemon
        self._hotkey = hotkey_listener

        self.menu = [
            rumps.MenuItem("Status: Ready", callback=None),
            None,  # separator
            rumps.MenuItem("Quit", callback=self._quit),
        ]
        self._status_item = self.menu["Status: Ready"]

        self._poll_timer = rumps.Timer(self._poll_state, 0.3)
        self._poll_timer.start()
        self._last_state = State.IDLE

    def _poll_state(self, _timer: rumps.Timer) -> None:
        state = self._daemon._state
        if state != self._last_state:
            self._last_state = state
            self.title = ICONS.get(state, "🎙")
            self._status_item.title = f"Status: {TITLES.get(state, 'Unknown')}"

    def _quit(self, _sender: rumps.MenuItem) -> None:
        logger.info("Quit from menu bar")
        self._hotkey.stop()
        self._daemon.shutdown()
        rumps.quit_application()


def run_with_menubar(daemon: object, hotkey_listener: object) -> None:
    """Run the daemon event loop in a background thread, menu bar on main thread."""
    daemon_thread = threading.Thread(target=daemon.run, daemon=True)
    daemon_thread.start()

    app = MenuBarApp(daemon, hotkey_listener)
    app.run()
