"""Menu bar status icon using raw pyobjc (NSStatusBar + NSMenu).

Replaces rumps which is broken on macOS 14+/Sequoia — menus don't drop down,
multiple phantom icons appear, and @clicked decorators silently fail.
"""

import concurrent.futures
import logging
import os
import queue
import re
import threading
import time
import wave
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy as np

import objc
from AppKit import (
    NSApplication,
    NSApplicationActivationPolicyAccessory,
    NSMenu,
    NSMenuItem,
    NSObject,
    NSStatusBar,
    NSTimer,
    NSVariableStatusItemLength,
)
from Foundation import NSRunLoop, NSRunLoopCommonModes
from PyObjCTools import AppHelper

from whisper_daemon import supervisor
from whisper_daemon.bridge_protocol import (
    apply_settings_patch,
    build_settings_message,
    build_status_message,
)
from whisper_daemon.config import VALID_FORMATS, Settings, load_settings, save_settings
from whisper_daemon.daemon import State

logger = logging.getLogger(__name__)

AUDIO_VIDEO_EXTENSIONS = [
    "mp3", "wav", "m4a", "flac", "ogg", "webm",
    "mp4", "mkv", "avi", "mov", "aac", "wma",
]

ICONS = {
    State.IDLE: "🎙",
    State.RECORDING: "🔴",
    State.TRANSCRIBING: "⏳",
}

MEETING_RECORDING_SYMBOL = "🔴"

# Cap on browser PCM staged before the meeting recorder exists (~2 min of
# 16kHz float32 mono). Bounds memory if the extension streams with no
# meeting starting.
BROWSER_PREBUFFER_MAX_BYTES = 8 * 1024 * 1024

TITLES = {
    State.IDLE: "Ready",
    State.RECORDING: "Recording...",
    State.TRANSCRIBING: "Transcribing...",
}


class MenuBarDelegate(NSObject):
    """NSApplication delegate that manages the status bar item and menu."""

    def initWithDaemon_hotkeyListener_(self, daemon, hotkey_listener):
        self = objc.super(MenuBarDelegate, self).init()
        if self is None:
            return None

        self._daemon = daemon
        self._hotkey = hotkey_listener
        self._meeting_active = False
        self._meeting_start = 0.0
        self._meeting_thread = None
        self._meeting_browser_triggered = False
        # Meeting lifecycle phase reported to the extension's control surface:
        # "idle" -> "recording" -> "transcribing" -> "idle". Distinct from
        # _meeting_active (which is False during the transcription tail) so a
        # get_settings arriving mid-transcription reports "transcribing", not
        # "idle", and the in-bar button stays correct until the real terminal
        # status lands.
        self._meeting_phase = "idle"
        self._meeting_chunk_count = 0
        # How many entries of the meeting's all_results list have already been
        # forwarded to the extension — captions are send-once + append (F5).
        self._results_sent_count = 0
        self._last_state = State.IDLE
        # Guards cross-thread access to self._settings: set_settings applies on
        # the bridge/worker thread while native menu toggles mutate on the main
        # thread (F14).
        self._settings_lock = threading.Lock()
        self._settings = load_settings()
        self._daemon._settings = self._settings

        # Browser audio bridge (Chrome extension)
        from whisper_daemon.audio_server import BrowserAudioBridge
        self._browser_bridge = BrowserAudioBridge(
            host=self._settings.server_host,
            port=self._settings.server_port,
            on_connect=self._on_browser_connect,
            on_audio=self._on_browser_audio,
            on_disconnect=self._on_browser_disconnect,
            on_control=self._on_browser_control,
        )
        self._browser_recorder = None  # set during meeting with browser source
        # Staging buffer for browser PCM that arrives after a reconnect's
        # 'start' but before the meeting worker has created the browser
        # recorder — otherwise the extension's flushed buffer is dropped.
        self._browser_prebuffer: list[bytes] = []
        self._browser_prebuffer_bytes = 0

        return self

    def applicationDidFinishLaunching_(self, notification):
        self._setup_status_bar()
        self._start_poll_timer()
        self._browser_bridge.start()

    def _setup_status_bar(self):
        status_bar = NSStatusBar.systemStatusBar()
        self._status_item = status_bar.statusItemWithLength_(NSVariableStatusItemLength)
        self._set_icon(State.IDLE)
        self._status_item.setHighlightMode_(True)

        menu = NSMenu.alloc().init()

        self._status_menu_item = _make_item("Status: Ready", None, self)
        self._status_menu_item.setEnabled_(False)
        menu.addItem_(self._status_menu_item)

        menu.addItem_(NSMenuItem.separatorItem())

        # Recent transcriptions submenu
        self._recent_menu_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Recent", None, ""
        )
        self._recent_menu = NSMenu.alloc().init()
        no_items = _make_item("No transcriptions yet", None, None)
        no_items.setEnabled_(False)
        self._recent_menu.addItem_(no_items)
        self._recent_menu_item.setSubmenu_(self._recent_menu)
        menu.addItem_(self._recent_menu_item)

        menu.addItem_(NSMenuItem.separatorItem())

        self._meeting_menu_item = _make_item(
            "Start Meeting Recording", "onMeeting:", self
        )
        menu.addItem_(self._meeting_menu_item)

        menu.addItem_(NSMenuItem.separatorItem())

        menu.addItem_(_make_item("Transcribe Files...", "onTranscribeFiles:", self))
        menu.addItem_(_make_item("Transcribe Folder...", "onTranscribeFolder:", self))

        menu.addItem_(NSMenuItem.separatorItem())

        # Settings submenu
        settings_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Settings", None, ""
        )
        settings_menu = NSMenu.alloc().init()

        # Recording Device (submenu with radio-style selection)
        rec_dev_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Recording Device", None, ""
        )
        self._rec_dev_menu = NSMenu.alloc().init()
        self._rec_dev_items = {}
        self._build_device_menu()
        rec_dev_item.setSubmenu_(self._rec_dev_menu)
        settings_menu.addItem_(rec_dev_item)

        # Save Audio toggle
        self._save_audio_item = _make_item(
            "Save Audio Recording", "onToggleSaveAudio:", self
        )
        if self._settings.save_audio:
            self._save_audio_item.setState_(1)
        settings_menu.addItem_(self._save_audio_item)

        # Capture Screenshots toggle
        self._capture_screenshots_item = _make_item(
            "Capture Screenshots", "onToggleScreenshots:", self
        )
        if self._settings.capture_screenshots:
            self._capture_screenshots_item.setState_(1)
        settings_menu.addItem_(self._capture_screenshots_item)

        # Speaker Diarization toggle
        self._diarize_item = _make_item(
            "Speaker Diarization", "onToggleDiarize:", self
        )
        if self._settings.diarize:
            self._diarize_item.setState_(1)
        settings_menu.addItem_(self._diarize_item)

        # Auto-Record Meetings toggle
        self._auto_record_item = _make_item(
            "Auto-Record Meetings", "onToggleAutoRecord:", self
        )
        if self._settings.auto_record_meetings:
            self._auto_record_item.setState_(1)
        settings_menu.addItem_(self._auto_record_item)

        # TTS Language (submenu with radio-style selection)
        tts_lang_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "TTS Language", None, ""
        )
        self._tts_lang_menu = NSMenu.alloc().init()
        self._tts_lang_items = {}
        for lang_code, lang_label in [("auto", "Auto-detect"), ("pl", "Polish"), ("en", "English")]:
            li = _make_item(lang_label, "onSelectTTSLang:", self)
            li.setRepresentedObject_(lang_code)
            if self._settings.tts_language == lang_code:
                li.setState_(1)
            self._tts_lang_menu.addItem_(li)
            self._tts_lang_items[lang_code] = li
        tts_lang_item.setSubmenu_(self._tts_lang_menu)
        settings_menu.addItem_(tts_lang_item)

        settings_menu.addItem_(NSMenuItem.separatorItem())

        # Start at Login toggle
        from whisper_daemon.autostart import is_enabled as autostart_enabled
        self._autostart_item = _make_item(
            "Start at Login", "onToggleAutostart:", self
        )
        if autostart_enabled():
            self._autostart_item.setState_(1)
        settings_menu.addItem_(self._autostart_item)

        settings_menu.addItem_(NSMenuItem.separatorItem())

        # Recording Folder
        self._rec_dir_item = _make_item(
            self._format_dir_label("Recording Folder", self._settings.recording_dir),
            "onChangeRecDir:", self,
        )
        settings_menu.addItem_(self._rec_dir_item)

        # Recording Format (submenu with checkmarks)
        rec_fmt_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Recording Format", None, ""
        )
        self._rec_fmt_menu = NSMenu.alloc().init()
        self._rec_fmt_items = {}
        for fmt in VALID_FORMATS:
            fi = _make_item(fmt, "onToggleRecFmt:", self)
            fi.setTag_(list(VALID_FORMATS).index(fmt))
            if fmt in self._settings.recording_formats:
                fi.setState_(1)  # NSOnState
            self._rec_fmt_menu.addItem_(fi)
            self._rec_fmt_items[fmt] = fi
        rec_fmt_item.setSubmenu_(self._rec_fmt_menu)
        settings_menu.addItem_(rec_fmt_item)

        settings_menu.addItem_(NSMenuItem.separatorItem())

        # Transcription Output Folder
        self._trans_dir_item = _make_item(
            self._format_dir_label(
                "Transcription Output",
                self._settings.transcription_output_dir or "same as input",
            ),
            "onChangeTransDir:", self,
        )
        settings_menu.addItem_(self._trans_dir_item)

        # Transcription Format (submenu with checkmarks)
        trans_fmt_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Transcription Format", None, ""
        )
        self._trans_fmt_menu = NSMenu.alloc().init()
        self._trans_fmt_items = {}
        for fmt in VALID_FORMATS:
            fi = _make_item(fmt, "onToggleTransFmt:", self)
            fi.setTag_(list(VALID_FORMATS).index(fmt))
            if fmt in self._settings.transcription_formats:
                fi.setState_(1)
            self._trans_fmt_menu.addItem_(fi)
            self._trans_fmt_items[fmt] = fi
        trans_fmt_item.setSubmenu_(self._trans_fmt_menu)
        settings_menu.addItem_(trans_fmt_item)

        settings_item.setSubmenu_(settings_menu)
        menu.addItem_(settings_item)

        menu.addItem_(NSMenuItem.separatorItem())

        menu.addItem_(_make_item("Quit", "onQuit:", self))

        self._status_item.setMenu_(menu)

    def _start_poll_timer(self):
        self._timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            0.3, self, "pollState:", None, True
        )
        NSRunLoop.currentRunLoop().addTimer_forMode_(
            self._timer, NSRunLoopCommonModes
        )

    # -- Poll timer --

    @objc.typedSelector(b"v@:@")
    def pollState_(self, timer):
        supervisor.beat("main-thread")
        if self._meeting_active:
            elapsed = time.monotonic() - self._meeting_start
            mins, secs = divmod(int(elapsed), 60)
            self._set_icon(State.RECORDING)
            self._meeting_menu_item.setTitle_(f"Stop Recording ({mins}:{secs:02d})")
            self._status_menu_item.setTitle_(f"Meeting recording ({mins}:{secs:02d})")
            return

        state = self._daemon._state
        if state != self._last_state:
            self._last_state = state
            self._set_icon(state)
            self._status_menu_item.setTitle_(
                f"Status: {TITLES.get(state, 'Unknown')}"
            )

        self._update_recent_menu()

    # -- Menu actions --

    @objc.typedSelector(b"v@:@")
    def onMeeting_(self, sender):
        if self._meeting_active:
            self._stop_meeting()
        else:
            self._start_meeting()

    @objc.typedSelector(b"v@:@")
    def onTranscribeFiles_(self, sender):
        from AppKit import NSOpenPanel

        panel = NSOpenPanel.openPanel()
        panel.setCanChooseFiles_(True)
        panel.setCanChooseDirectories_(False)
        panel.setAllowsMultipleSelection_(True)
        panel.setAllowedFileTypes_(AUDIO_VIDEO_EXTENSIONS)
        panel.setTitle_("Select audio/video files to transcribe")

        if panel.runModal() != 1:
            return

        paths = [str(url.path()) for url in panel.URLs()]
        if paths:
            threading.Thread(
                target=self._transcribe_paths_worker, args=(paths,), daemon=True
            ).start()

    @objc.typedSelector(b"v@:@")
    def onTranscribeFolder_(self, sender):
        from AppKit import NSOpenPanel

        panel = NSOpenPanel.openPanel()
        panel.setCanChooseFiles_(False)
        panel.setCanChooseDirectories_(True)
        panel.setAllowsMultipleSelection_(False)
        panel.setTitle_("Select folder to transcribe")

        if panel.runModal() != 1:
            return

        folder = str(panel.URLs()[0].path())
        if folder:
            threading.Thread(
                target=self._transcribe_paths_worker, args=([folder],), daemon=True
            ).start()

    @objc.typedSelector(b"v@:@")
    def onQuit_(self, sender):
        """Quit gracefully: finish any in-flight meeting save FIRST.

        AppHelper.stopEventLoop() ends in NSApp.terminate_, which exits the
        process immediately — so it must be the very last step, called only
        after the meeting worker has saved its transcript. The shutdown work
        happens on a helper thread; the exit watchdog guarantees the process
        dies even if that thread wedges (long timeout while a meeting save
        is running, short otherwise).
        """
        if getattr(self, "_quitting", False):
            return
        self._quitting = True
        logger.info("Quit from menu bar")

        supervisor.suspend("quit from menu bar")
        meeting_active = self._meeting_active
        _start_exit_watchdog(330.0 if meeting_active else 15.0, exit_code=0)

        self._hotkey.stop()
        if hasattr(self, "_timer") and self._timer is not None:
            self._timer.invalidate()
            self._timer = None
        self._set_icon_by_name("⏳")

        threading.Thread(
            target=self._shutdown_worker, name="quit-shutdown", daemon=True
        ).start()

    def _shutdown_worker(self) -> None:
        try:
            self.graceful_stop()
            self._daemon.shutdown()
        except Exception:
            logger.exception("Error during shutdown")
        finally:
            AppHelper.callAfter(AppHelper.stopEventLoop)

    def _set_icon(self, state: State) -> None:
        """Set the menu bar icon emoji for the given state (main thread only)."""
        self._status_item.button().setTitle_(ICONS.get(state, "🎙"))

    def _set_icon_by_name(self, icon_text: str, _fallback: str = "") -> None:
        """Set the menu bar icon to the given text (main thread only)."""
        self._status_item.button().setTitle_(icon_text)

    def _set_icon_safe(self, state: State) -> None:
        """Set the menu bar icon from any thread."""
        AppHelper.callAfter(self._set_icon, state)

    def _set_status_safe(self, text: str) -> None:
        """Set the status menu item title from any thread."""
        AppHelper.callAfter(self._status_menu_item.setTitle_, text)

    # -- Recent transcriptions --

    def _update_recent_menu(self):
        history = self._daemon.history
        if not history:
            return

        current_count = self._recent_menu.numberOfItems()
        if current_count == len(history):
            # Check if first item matches — if so, no update needed
            first = self._recent_menu.itemAtIndex_(0)
            if first and str(first.title()).startswith(history[0][:30]):
                return

        self._recent_menu.removeAllItems()
        for i, text in enumerate(history):
            truncated = text[:50] + "..." if len(text) > 50 else text
            truncated = truncated.replace("\n", " ")
            item = _make_item(f"\"{truncated}\"", "onCopyRecent:", self)
            item.setTag_(i)
            self._recent_menu.addItem_(item)

    @objc.typedSelector(b"v@:@")
    def onCopyRecent_(self, sender):
        idx = sender.tag()
        history = self._daemon.history
        if idx < len(history):
            from AppKit import NSPasteboard, NSPasteboardTypeString
            pb = NSPasteboard.generalPasteboard()
            pb.clearContents()
            pb.setString_forType_(history[idx], NSPasteboardTypeString)
            logger.info("Copied transcription #%d to clipboard (%d chars)", idx + 1, len(history[idx]))

    # -- Settings actions --

    @objc.typedSelector(b"v@:@")
    def onToggleSaveAudio_(self, sender):
        with self._settings_lock:
            self._settings.save_audio = not self._settings.save_audio
        sender.setState_(1 if self._settings.save_audio else 0)
        save_settings(self._settings)
        logger.info("Save audio: %s", self._settings.save_audio)

    @objc.typedSelector(b"v@:@")
    def onToggleAutostart_(self, sender):
        from whisper_daemon.autostart import disable, enable, is_enabled
        if is_enabled():
            disable()
            sender.setState_(0)
        else:
            enable()
            sender.setState_(1)

    @objc.typedSelector(b"v@:@")
    def onToggleScreenshots_(self, sender):
        with self._settings_lock:
            self._settings.capture_screenshots = not self._settings.capture_screenshots
        sender.setState_(1 if self._settings.capture_screenshots else 0)
        save_settings(self._settings)
        logger.info("Capture screenshots: %s", self._settings.capture_screenshots)

    @objc.typedSelector(b"v@:@")
    def onToggleDiarize_(self, sender):
        with self._settings_lock:
            self._settings.diarize = not self._settings.diarize
        sender.setState_(1 if self._settings.diarize else 0)
        save_settings(self._settings)
        logger.info("Speaker diarization: %s", self._settings.diarize)

    @objc.typedSelector(b"v@:@")
    def onToggleAutoRecord_(self, sender):
        with self._settings_lock:
            self._settings.auto_record_meetings = not self._settings.auto_record_meetings
        sender.setState_(1 if self._settings.auto_record_meetings else 0)
        save_settings(self._settings)
        logger.info("Auto-record meetings: %s", self._settings.auto_record_meetings)

    @objc.typedSelector(b"v@:@")
    def onSelectTTSLang_(self, sender):
        lang_code = str(sender.representedObject())
        for item in self._tts_lang_items.values():
            item.setState_(0)
        sender.setState_(1)
        with self._settings_lock:
            self._settings.tts_language = lang_code
        save_settings(self._settings)
        logger.info("TTS language changed to: %s", lang_code)

    @objc.typedSelector(b"v@:@")
    def onChangeRecDir_(self, sender):
        from AppKit import NSOpenPanel

        panel = NSOpenPanel.openPanel()
        panel.setCanChooseFiles_(False)
        panel.setCanChooseDirectories_(True)
        panel.setAllowsMultipleSelection_(False)
        panel.setTitle_("Choose recording output folder")

        if panel.runModal() != 1:
            return

        path = str(panel.URLs()[0].path())
        with self._settings_lock:
            self._settings.recording_dir = path
        save_settings(self._settings)
        self._rec_dir_item.setTitle_(
            self._format_dir_label("Recording Folder", path)
        )

    @objc.typedSelector(b"v@:@")
    def onToggleRecFmt_(self, sender):
        fmt = str(sender.title())
        with self._settings_lock:
            if fmt in self._settings.recording_formats:
                if len(self._settings.recording_formats) > 1:
                    self._settings.recording_formats.remove(fmt)
                    sender.setState_(0)
            else:
                self._settings.recording_formats.append(fmt)
                sender.setState_(1)
        save_settings(self._settings)

    @objc.typedSelector(b"v@:@")
    def onChangeTransDir_(self, sender):
        from AppKit import NSOpenPanel

        panel = NSOpenPanel.openPanel()
        panel.setCanChooseFiles_(False)
        panel.setCanChooseDirectories_(True)
        panel.setAllowsMultipleSelection_(False)
        panel.setTitle_("Choose transcription output folder")

        if panel.runModal() != 1:
            return

        path = str(panel.URLs()[0].path())
        with self._settings_lock:
            self._settings.transcription_output_dir = path
        save_settings(self._settings)
        self._trans_dir_item.setTitle_(
            self._format_dir_label("Transcription Output", path)
        )

    @objc.typedSelector(b"v@:@")
    def onToggleTransFmt_(self, sender):
        fmt = str(sender.title())
        with self._settings_lock:
            if fmt in self._settings.transcription_formats:
                if len(self._settings.transcription_formats) > 1:
                    self._settings.transcription_formats.remove(fmt)
                    sender.setState_(0)
            else:
                self._settings.transcription_formats.append(fmt)
                sender.setState_(1)
        save_settings(self._settings)

    def _build_device_menu(self):
        """Populate the recording device submenu with available input devices."""
        try:
            import sounddevice as sd

            self._rec_dev_menu.removeAllItems()
            self._rec_dev_items = {}

            # Default device option
            default_item = _make_item("System Default", "onSelectDevice:", self)
            default_item.setTag_(0)
            if not self._settings.recording_device:
                default_item.setState_(1)
            self._rec_dev_menu.addItem_(default_item)
            self._rec_dev_items[""] = default_item

            self._rec_dev_menu.addItem_(NSMenuItem.separatorItem())

            devs = sd.query_devices()
            tag = 1
            for i, d in enumerate(devs):
                if d["max_input_channels"] > 0:
                    name = d["name"]
                    item = _make_item(name, "onSelectDevice:", self)
                    item.setTag_(tag)
                    tag += 1
                    if name == self._settings.recording_device:
                        item.setState_(1)
                    self._rec_dev_menu.addItem_(item)
                    self._rec_dev_items[name] = item
        except Exception:
            logger.exception("Failed to build device menu")

    @objc.typedSelector(b"v@:@")
    def onSelectDevice_(self, sender):
        title = str(sender.title())
        device_name = "" if title == "System Default" else title

        for item in self._rec_dev_items.values():
            item.setState_(0)
        sender.setState_(1)

        with self._settings_lock:
            self._settings.recording_device = device_name
        save_settings(self._settings)
        logger.info("Recording device changed to: %s", device_name or "system default")

    @staticmethod
    def _format_dir_label(prefix: str, path: str) -> str:
        display = path.replace(str(Path.home()), "~") if path else "same as input"
        return f"{prefix}: {display}"

    # -- Browser bridge callbacks (called from asyncio thread) --

    def _on_browser_connect(self, title: str, url: str) -> None:
        """Extension started a capture session.

        A {"type":"start"} from the extension is always an explicit user
        gesture now (it follows the getDisplayMedia permission click), so it
        must always start the meeting when none is active — regardless of the
        auto_record_meetings preference, which only governs fully-automatic
        starts and is intentionally not consulted here.
        """
        logger.info("Browser meeting started: %s (%s)", title, url)
        if not self._meeting_active:
            AppHelper.callAfter(self._start_meeting, True, title)

    def _on_browser_audio(self, data: bytes) -> None:
        """Raw PCM audio from browser tab."""
        rec = self._browser_recorder
        if rec is not None:
            rec.feed_audio(np.frombuffer(data, dtype=np.float32))
            return
        # No recorder yet: stage a bounded amount so the extension's flushed
        # reconnect buffer isn't lost while the meeting worker spins up. The
        # worker drains this when it creates the browser recorder; if no
        # meeting starts it's cleared on the next reset. Drop-oldest keeps
        # recent speech.
        self._browser_prebuffer.append(data)
        self._browser_prebuffer_bytes += len(data)
        while self._browser_prebuffer_bytes > BROWSER_PREBUFFER_MAX_BYTES and self._browser_prebuffer:
            self._browser_prebuffer_bytes -= len(self._browser_prebuffer.pop(0))

    def _on_browser_disconnect(self) -> None:
        """Extension disconnected or stopped capture."""
        logger.info("Browser meeting ended")
        if self._meeting_active and self._meeting_browser_triggered:
            AppHelper.callAfter(self._stop_meeting)

    def _list_input_devices(self) -> list[str]:
        """Return the names of available input devices (best-effort)."""
        try:
            import sounddevice as sd

            return [
                d["name"]
                for d in sd.query_devices()
                if d["max_input_channels"] > 0
            ]
        except Exception:
            logger.exception("Failed to query input devices")
            return []

    def _push_settings_snapshot(self) -> None:
        """Send the current settings + device list to the extension."""
        try:
            msg = build_settings_message(
                self._settings, self._list_input_devices()
            )
            self._browser_bridge.send_json_str(msg)
        except Exception:
            logger.exception("Failed to push settings snapshot")

    def _run_off_bridge(self, fn: Callable[[], None]) -> None:
        """Run blocking work off the bridge event-loop thread.

        Device enumeration (sounddevice) and disk I/O must never run inline on
        the WebSocket coroutine thread — while they block, the async-for loop
        can't read the next audio frame, silently starving a live meeting.
        """
        threading.Thread(target=fn, daemon=True).start()

    def _handle_get_settings(self) -> None:
        """Push settings snapshot + current phase status (worker thread).

        Reports the daemon's real phase (idle/recording/transcribing), not a
        hardcoded "idle": a control surface that connects while the previous
        meeting is still transcribing must see "transcribing" so its in-bar
        button doesn't reset early. Plain attribute reads — thread-safe.
        """
        self._push_settings_snapshot()
        phase = self._meeting_phase
        elapsed = (
            time.monotonic() - self._meeting_start if phase == "recording" else 0.0
        )
        try:
            self._browser_bridge.send_json_str(
                build_status_message(phase, elapsed, self._meeting_chunk_count)
            )
        except Exception:
            logger.exception("Failed to push status")

    def _persist_and_push_settings(self) -> None:
        """Persist settings to disk and echo the snapshot (worker thread)."""
        try:
            save_settings(self._settings)
        except Exception:
            logger.exception("Failed to persist settings from extension")
        self._push_settings_snapshot()

    def _on_browser_control(self, data: dict) -> None:
        """Handle control messages (get_settings/set_settings/stop_meeting).

        Runs on the bridge asyncio thread. Blocking work (device enumeration,
        disk I/O) is offloaded to a worker thread so the audio loop never
        stalls; AppKit-touching work is dispatched to the main thread.
        """
        msg_type = data.get("type")

        if msg_type == "get_settings":
            # _push_settings_snapshot enumerates PortAudio devices (blocking);
            # never run it on the bridge loop or audio reception stalls.
            self._run_off_bridge(self._handle_get_settings)

        elif msg_type == "set_settings":
            patch = data.get("patch")
            if not isinstance(patch, dict):
                logger.warning("set_settings without a dict patch — ignoring")
                return
            # Immutable in-memory update is fast; do it inline so ordering is
            # deterministic. Disk persistence + device enumeration are offloaded.
            # Locked so a concurrent native menu toggle (main thread) can't
            # interleave its read-modify-write with this one (F14).
            with self._settings_lock:
                self._settings = apply_settings_patch(self._settings, patch)
                self._daemon._settings = self._settings
            # COLD settings (mic/formats/dir/diarize/displays/capture_mic/tab)
            # take effect on the next recording — no live swap attempted. Hot
            # settings (live_captions is UI-only; capture_screenshots' local
            # capturer isn't reachable from here) also apply next recording.
            self._run_off_bridge(self._persist_and_push_settings)
            # Keep the native menu bar checkmarks/labels in sync (main thread).
            AppHelper.callAfter(self._sync_menu_from_settings)

        elif msg_type == "stop_meeting":
            if self._meeting_active:
                AppHelper.callAfter(self._stop_meeting)

    def _sync_menu_from_settings(self) -> None:
        """Re-apply menu checkmarks/labels from self._settings (main thread).

        Called after a set_settings patch from the extension so the native
        macOS menu doesn't disagree with the daemon's actual settings.
        """
        s = self._settings
        try:
            self._save_audio_item.setState_(1 if s.save_audio else 0)
            self._capture_screenshots_item.setState_(
                1 if s.capture_screenshots else 0
            )
            self._diarize_item.setState_(1 if s.diarize else 0)
            self._auto_record_item.setState_(1 if s.auto_record_meetings else 0)
            self._rec_dir_item.setTitle_(
                self._format_dir_label("Recording Folder", s.recording_dir)
            )
            for fmt, item in self._rec_fmt_items.items():
                item.setState_(1 if fmt in s.recording_formats else 0)
            # Device list itself is unchanged by a patch; only re-mark the
            # selected device to avoid a blocking re-query on the UI thread.
            for name, item in self._rec_dev_items.items():
                item.setState_(1 if name == s.recording_device else 0)
        except Exception:
            logger.exception("Failed to sync menu from settings")

    # -- Meeting recording --

    def _start_meeting(self, browser_triggered: bool = False, browser_title: str = "") -> None:
        # Guard against duplicate starts: two rapid browser 'start' messages
        # each schedule _start_meeting via callAfter, and _on_browser_connect
        # checks _meeting_active before the first one has run — so the check
        # there is not enough. A second concurrent meeting would open a
        # second mic stream and corrupt the chunk pipeline.
        if self._meeting_active:
            logger.info("Meeting already active — ignoring duplicate start")
            return
        self._meeting_active = True
        self._meeting_browser_triggered = browser_triggered
        self._meeting_start = time.monotonic()
        self._meeting_phase = "recording"
        self._meeting_chunk_count = 0
        self._results_sent_count = 0
        self._meeting_menu_item.setTitle_("Stop Recording (0:00)")
        self._set_icon_by_name(MEETING_RECORDING_SYMBOL)

        trigger = "browser" if browser_triggered else "menu bar"
        logger.info("Meeting recording started from %s", trigger)

        self._meeting_thread = threading.Thread(
            target=self._meeting_worker,
            args=(browser_triggered, browser_title),
            daemon=False,
        )
        self._meeting_thread.start()

    def _stop_meeting(self) -> None:
        if not self._meeting_active:
            # Second stop for the same meeting (e.g. browser 'stop' message
            # plus the connection-closed callback) — nothing to do.
            return
        self._meeting_active = False
        logger.info("Meeting recording stop requested")

        # If the worker thread never started or already finished, reset UI immediately
        if self._meeting_thread is None or not self._meeting_thread.is_alive():
            logger.warning("Meeting worker not running — resetting UI directly")
            self._reset_meeting_ui()
            return

        self._meeting_menu_item.setTitle_("Start Meeting Recording")
        self._set_icon(State.TRANSCRIBING)
        self._status_menu_item.setTitle_("Finishing transcription...")

    def _meeting_worker(self, browser_triggered: bool = False, browser_title: str = "") -> None:
        from whisper_daemon import telemetry
        from whisper_daemon.meeting_recorder import AudioChunk, MeetingRecorder
        from whisper_daemon.screen_capture import ScreenCapture
        from whisper_daemon.transcriber import transcribe_full

        model = self._daemon._model
        device = self._settings.recording_device or None
        chunk_queue: queue.Queue[AudioChunk | None] = queue.Queue()

        # Which sources to capture (settings snapshot; may be toggled from the
        # extension between meetings — cold, so read here at meeting start).
        capture_mic = self._settings.capture_mic
        capture_tab = self._settings.capture_tab

        # Snapshot the remaining COLD settings the worker consumes, so a
        # mid-meeting set_settings from the extension cannot alter this
        # in-progress meeting's output (contract section D).
        diarize = self._settings.diarize
        save_audio = self._settings.save_audio
        recording_formats = list(self._settings.recording_formats)
        recording_dir_path = self._settings.recording_dir_path
        capture_screenshots = self._settings.capture_screenshots
        screenshot_displays = self._settings.screenshot_displays
        screenshot_interval = self._settings.screenshot_interval
        # "auto" -> None so Whisper auto-detects; otherwise force the language so
        # short/quiet chunks stop coming back as the wrong language (Icelandic,
        # Russian, …) in the live captions and saved transcript.
        language = (
            None if self._settings.recording_language == "auto"
            else self._settings.recording_language
        )

        # Mic recorder (local device) — only when capture_mic is enabled.
        mic_recorder: MeetingRecorder | None = None
        if capture_mic:
            mic_recorder = MeetingRecorder(chunk_queue, device=device, source_label="mic")

        # Browser recorder (Chrome extension tab audio) — only when capture_tab
        # is enabled AND a browser-triggered connection is live.
        browser_recorder: MeetingRecorder | None = None
        if capture_tab and browser_triggered and self._browser_bridge.connected:
            browser_recorder = MeetingRecorder(chunk_queue, source_label="browser")
            browser_recorder.start_without_device()
            # Drain PCM staged between the extension's 'start' and now so a
            # reconnect's flushed buffer (or the first frames of a fresh
            # session) isn't lost.
            staged = self._browser_prebuffer
            self._browser_prebuffer = []
            self._browser_prebuffer_bytes = 0
            for data in staged:
                browser_recorder.feed_audio(np.frombuffer(data, dtype=np.float32))
            self._browser_recorder = browser_recorder

        # No source at all: nothing to record — abort cleanly rather than
        # spinning an empty meeting that never produces chunks.
        if mic_recorder is None and browser_recorder is None:
            logger.warning(
                "No audio source (capture_mic off and no browser recorder) — "
                "aborting meeting"
            )
            _notify(
                "whisper-daemon",
                "No audio source",
                "Enable microphone or participant capture — nothing to record.",
            )
            self._reset_meeting_ui()
            return

        # Number of recorders determines how many None sentinels to expect
        sentinel_expected = (1 if mic_recorder else 0) + (1 if browser_recorder else 0)
        sentinel_count = 0

        all_results: list[tuple[float, dict]] = []
        all_audio: list[np.ndarray] = []
        chunk_count = 0

        # Prepare output dir early for screenshots. The meeting title is
        # untrusted (comes from the extension's 'start' message), so strip it
        # to a safe filename component — keep only [A-Za-z0-9_-] to prevent
        # path traversal (e.g. "../../tmp/pwn") out of recording_dir.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        title_slug = (
            re.sub(r"[^A-Za-z0-9_-]", "", browser_title.replace(" ", "_"))[:30]
            if browser_title
            else ""
        )
        dir_name = f"recording_{title_slug}_{timestamp}" if title_slug else f"recording_{timestamp}"
        base_dir = recording_dir_path
        rec_dir = base_dir / dir_name
        # Defense in depth: ensure the resolved dir stays under the configured
        # recording folder even if the slug logic is ever weakened.
        if not rec_dir.resolve().is_relative_to(base_dir.resolve()):
            logger.warning("Meeting dir escaped recording folder — using base dir")
            rec_dir = base_dir / f"recording_{timestamp}"
        rec_dir.mkdir(parents=True, exist_ok=True)

        screen_capture: ScreenCapture | None = None
        if capture_screenshots:
            screen_capture = ScreenCapture(
                rec_dir,
                interval=screenshot_interval,
                displays=screenshot_displays,
            )
            screen_capture.start()

        HEALTH_CHECK_SEC = 120.0  # first warning after 2 min of silence
        HEALTH_REPEAT_SEC = 120.0  # re-warn every 2 min while still silent
        RECOVERY_BACKOFF_SEC = 10.0  # min interval between recovery attempts

        telemetry.meeting_start()
        devices_stopped = False
        if mic_recorder is not None:
            try:
                mic_recorder.start()
            except Exception as exc:
                logger.error("Failed to open mic for meeting: %s", exc)
                # If the browser tab is still feeding audio, keep the meeting
                # alive on that source alone instead of aborting.
                if browser_recorder is None:
                    if screen_capture:
                        screen_capture.stop()
                    _notify(
                        "whisper-daemon",
                        "Error",
                        "Could not open microphone for meeting.",
                    )
                    self._reset_meeting_ui()
                    return
                logger.warning("Mic failed to open — continuing with tab audio only")
                mic_recorder = None
                sentinel_expected = 1 if browser_recorder else 0

        if mic_recorder is not None and mic_recorder.fell_back_to_default:
            _notify(
                "whisper-daemon",
                "Microphone fallback",
                "Preferred device unavailable — using system default mic.",
            )

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                futures: dict[concurrent.futures.Future, float] = {}

                partial_path = rec_dir / "transcript_live.txt"
                # Mic liveness is read from the recorder's callback clock
                # (last_audio_at), not from emitted chunks — steady browser
                # audio no longer masks a dead microphone.
                last_health_warn = 0.0
                last_recovery_attempt = 0.0
                last_status_push = 0.0
                mic_lost_notified = False

                while self._meeting_active:
                    now = time.monotonic()

                    # Push a recording status to the extension about once/sec so
                    # the in-bar button + timer track the daemon (single source
                    # of truth) even for menu-bar-started meetings.
                    if now - last_status_push >= 1.0:
                        last_status_push = now
                        elapsed = now - self._meeting_start
                        self._meeting_chunk_count = chunk_count
                        try:
                            self._browser_bridge.send_json_str(
                                build_status_message("recording", elapsed, chunk_count)
                            )
                        except Exception:
                            logger.debug("Failed to push recording status", exc_info=True)

                    # Check if mic recorder needs device recovery. Retries
                    # keep firing on a backoff timer until they succeed or
                    # the meeting ends — devices (Bluetooth, USB) can
                    # reappear minutes later, so we never give up.
                    if (
                        mic_recorder is not None
                        and mic_recorder.needs_recovery
                        and now - last_recovery_attempt >= RECOVERY_BACKOFF_SEC
                    ):
                        last_recovery_attempt = now
                        logger.warning("Mic device needs recovery, attempting reopen...")
                        if mic_recorder.recover_device():
                            logger.info("Mic recovered, meeting continues")
                            mic_lost_notified = False
                            if mic_recorder.fell_back_to_default:
                                _notify(
                                    "whisper-daemon",
                                    "Mic recovered (fallback)",
                                    "Original device lost — now using system default mic.",
                                )
                        elif not mic_lost_notified:
                            # Notify once per lost-then-recovered cycle to
                            # avoid hammering the user during extended outages.
                            mic_lost_notified = True
                            logger.error("Mic recovery failed, will keep retrying every %.0fs", RECOVERY_BACKOFF_SEC)
                            _notify(
                                "whisper-daemon",
                                "Mic lost",
                                "Device recovery failed. Will keep retrying; browser audio continues.",
                            )

                    try:
                        chunk = chunk_queue.get(timeout=0.5)
                    except queue.Empty:
                        if _collect_futures(futures, all_results):
                            _write_partial(partial_path, all_results)
                            self._send_results_to_browser(all_results, chunk_count)
                        # Health check keyed on the mic CALLBACK, not on
                        # emitted chunks: a quiet room emits no chunks (VAD
                        # drops silence) yet the callback keeps firing, so
                        # keying on chunks would false-alarm every 2 min
                        # while the user is merely listening. last_audio_at
                        # only goes stale when the stream genuinely stops
                        # delivering samples — a real dead device.
                        mic_last_audio = (
                            mic_recorder.last_audio_at if mic_recorder is not None else 0.0
                        )
                        elapsed_silent = now - mic_last_audio if mic_last_audio else 0.0
                        if (
                            mic_recorder is not None
                            and elapsed_silent > HEALTH_CHECK_SEC
                            and now - last_health_warn >= HEALTH_REPEAT_SEC
                        ):
                            last_health_warn = now
                            logger.warning(
                                "No mic audio callbacks for %.0fs — mic stream appears dead",
                                elapsed_silent,
                            )
                            _notify(
                                "whisper-daemon",
                                "Microphone stopped",
                                f"No mic audio for {int(elapsed_silent)}s. Attempting to recover.",
                            )
                            mic_recorder.request_recovery()
                        continue

                    if chunk is None:
                        sentinel_count += 1
                        if sentinel_count >= sentinel_expected:
                            break
                        continue

                    chunk_count += 1
                    logger.info(
                        "Meeting chunk %d: %.1fs [%s]",
                        chunk_count, chunk.duration, chunk.source,
                    )
                    telemetry.meeting_chunk_queued(chunk_count, chunk.duration, chunk.start_time)
                    if save_audio or diarize:
                        all_audio.append(chunk.audio.copy())
                    cn = chunk_count  # capture for closure
                    def _transcribe_and_track(audio, m, n):
                        result = transcribe_full(audio, m, language)
                        segs = len(result.get("segments", []))
                        chars = len(result.get("text", ""))
                        telemetry.meeting_chunk_transcribed(n, chars, segs)
                        return result
                    future = pool.submit(_transcribe_and_track, chunk.audio, model, cn)
                    futures[future] = chunk.start_time
                    if _collect_futures(futures, all_results):
                        _write_partial(partial_path, all_results)
                        self._send_results_to_browser(all_results, chunk_count)

                # Recording finished — enter the transcription phase so a
                # get_settings arriving now reports "transcribing" (not idle),
                # and tell the extension so its button reflects that state.
                self._meeting_phase = "transcribing"
                self._meeting_chunk_count = chunk_count
                try:
                    self._browser_bridge.send_json_str(
                        build_status_message("transcribing", 0.0, chunk_count)
                    )
                except Exception:
                    logger.debug("Failed to push transcribing status", exc_info=True)

                if mic_recorder is not None:
                    mic_recorder.stop()
                if browser_recorder is not None:
                    browser_recorder.stop_without_device()
                    self._browser_recorder = None
                if screen_capture is not None:
                    screen_capture.stop()
                devices_stopped = True

                # Wait for in-flight transcription to finish FIRST
                _collect_futures(futures, all_results, wait=True)

                # Now safe to transcribe remaining chunks (no concurrent GPU access)
                while True:
                    try:
                        chunk = chunk_queue.get_nowait()
                    except queue.Empty:
                        break
                    if chunk is None:
                        continue
                    chunk_count += 1
                    if save_audio or diarize:
                        all_audio.append(chunk.audio.copy())
                    result = transcribe_full(chunk.audio, model, language)
                    if result.get("text", "").strip():
                        all_results.append((chunk.start_time, result))

        except Exception:
            logger.exception("Meeting recording failed")
            _notify("whisper-daemon", "Error", "Meeting recording failed.")
            self._reset_meeting_ui()
            return
        finally:
            # Safety net for exception paths that skipped the normal
            # cleanup: a leaked registered mic stream permanently disables
            # the PortAudio device refresh for the rest of the process life.
            # Skipped after a clean stop() (devices_stopped=True).
            if not devices_stopped:
                if mic_recorder is not None:
                    mic_recorder.abort()
                if browser_recorder is not None:
                    browser_recorder.stop_without_device()
                    self._browser_recorder = None
                if screen_capture is not None:
                    screen_capture.stop()

        if not all_results:
            _notify("whisper-daemon", "Done", "No speech detected.")
            self._reset_meeting_ui()
            return

        # Finalization runs OUTSIDE the worker's try/except above, so any
        # failure here (disk full, unwritable recording_dir, malformed
        # segments) must NOT leave _meeting_phase stuck at "transcribing".
        # The finally guarantees _reset_meeting_ui() runs and pushes the
        # terminal "idle" status over the still-open socket (F3).
        try:
            from whisper_daemon.formats import merge_chunk_results
            merged_result = merge_chunk_results(all_results)

            if diarize and all_audio:
                try:
                    from whisper_daemon.diarizer import diarize_batch
                    from whisper_daemon.diarize_merge import merge_speakers_with_transcript

                    full_audio = np.concatenate(all_audio)
                    logger.info("Diarizing %.1fs of audio...", len(full_audio) / 16000)
                    speaker_segments = diarize_batch(full_audio)
                    merged_result = merge_speakers_with_transcript(
                        speaker_segments, merged_result
                    )
                    speaker_count = len(merged_result.get("speakers", []))
                    logger.info("Diarization done — %d speakers", speaker_count)
                except Exception:
                    logger.exception("Diarization failed, saving without speaker labels")

            from whisper_daemon.formats import FORMATTERS

            written: list[str] = []
            for fmt in recording_formats:
                if fmt in FORMATTERS:
                    out = rec_dir / f"transcript.{fmt}"
                    out.write_text(FORMATTERS[fmt](merged_result), encoding="utf-8")
                    written.append(str(out))

            if save_audio and all_audio:
                audio_path = rec_dir / "recording.wav"
                full_audio = np.concatenate(all_audio)
                _save_wav(audio_path, full_audio, 16000)
                written.append(str(audio_path))

            screenshots_msg = ""
            if screen_capture is not None and screen_capture.saved_count > 0:
                screenshots_msg = f", {screen_capture.saved_count} screenshots"

            # Remove live partial now that final transcript exists
            partial_path = rec_dir / "transcript_live.txt"
            partial_path.unlink(missing_ok=True)

            telemetry.meeting_stop(chunk_count, str(rec_dir))
            logger.info("Meeting saved: %s", ", ".join(written))
            _notify(
                "whisper-daemon",
                f"Meeting recorded ({chunk_count} chunks{screenshots_msg})",
                str(rec_dir),
            )
        except Exception:
            logger.exception("Meeting finalization failed")
            _notify("whisper-daemon", "Error", "Failed to save meeting transcript.")
        finally:
            self._reset_meeting_ui()

    def _send_results_to_browser(self, all_results: list, chunk_count: int) -> None:
        """Forward NEW transcription results to the extension (send-once).

        Each result is sent exactly once, tagged with its true index in
        all_results as chunk_index — the old code resent all_results[-3:]
        every cycle, duplicating and reordering captions (F5). ``chunk_count``
        is unused for indexing now; kept for signature stability.
        """
        if not self._browser_bridge.connected:
            return
        from whisper_daemon.bridge_protocol import select_unsent_results

        for idx, start_time, text in select_unsent_results(
            all_results, self._results_sent_count
        ):
            self._browser_bridge.send_chunk_result(text, start_time, idx)
        self._results_sent_count = len(all_results)

    def graceful_stop(self, timeout: float = 300.0) -> None:
        """Stop any active meeting, wait for save/diarize, then quit.

        Called from signal handlers to ensure the meeting is properly saved
        before the process exits. Runs on the signal-handler thread — must
        not touch AppKit.
        """
        if self._meeting_active:
            logger.info("Graceful stop: stopping active meeting before exit")
            self._meeting_active = False
        # Join whenever the worker is alive — not only when _meeting_active.
        # If the user already clicked Stop, _meeting_active is False but the
        # worker is still saving/diarizing; skipping the join here would let
        # the exit path kill the save mid-write.
        if self._meeting_thread is not None and self._meeting_thread.is_alive():
            logger.info("Graceful stop: waiting for meeting save to finish")
            self._meeting_thread.join(timeout=timeout)
            if self._meeting_thread.is_alive():
                logger.warning("Meeting worker did not finish within %.0fs", timeout)
        self._browser_bridge.stop()

    def _reset_meeting_ui(self) -> None:
        """Reset meeting state and UI. Safe to call from any thread —
        AppKit mutations are dispatched to the main thread."""
        self._meeting_active = False
        self._meeting_browser_triggered = False
        self._meeting_phase = "idle"
        self._browser_recorder = None
        self._browser_prebuffer = []
        self._browser_prebuffer_bytes = 0

        # Tell the extension the meeting is over so its in-bar button goes idle.
        try:
            self._browser_bridge.send_json_str(
                build_status_message("idle", 0.0, 0)
            )
        except Exception:
            logger.debug("Failed to push idle status", exc_info=True)

        def _update_ui() -> None:
            self._meeting_menu_item.setTitle_("Start Meeting Recording")
            self._set_icon(State.IDLE)
            self._status_menu_item.setTitle_("Status: Ready")

        AppHelper.callAfter(_update_ui)

    # -- File transcription --

    def _transcribe_paths_worker(self, paths: list[str]) -> None:
        from whisper_daemon.formats import FORMATTERS
        from whisper_daemon.transcriber import transcribe_file

        model = self._daemon._model

        self._set_icon_safe(State.TRANSCRIBING)
        self._set_status_safe("Transcribing files...")

        files: list[Path] = []
        ext_set = {"." + e for e in AUDIO_VIDEO_EXTENSIONS}
        for p in paths:
            path = Path(p)
            if path.is_file():
                files.append(path)
            elif path.is_dir():
                for child in sorted(path.iterdir()):
                    if child.is_file() and child.suffix.lower() in ext_set:
                        files.append(child)

        if not files:
            _notify("whisper-daemon", "No files", "No audio/video files found.")
            self._reset_meeting_ui()
            return

        out_dir = self._settings.transcription_output_dir_path
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)

        done = 0
        last_error: str | None = None
        for file_path in files:
            try:
                self._set_status_safe(f"Transcribing {file_path.name}...")
                result = transcribe_file(str(file_path), model=model)

                dest = out_dir or file_path.parent
                stem = file_path.stem
                for fmt in self._settings.transcription_formats:
                    if fmt in FORMATTERS:
                        output = dest / f"{stem}.{fmt}"
                        output.write_text(
                            FORMATTERS[fmt](result), encoding="utf-8"
                        )
                done += 1
            except Exception as exc:
                last_error = str(exc) or exc.__class__.__name__
                logger.exception("Failed to transcribe %s", file_path)

        # Surface the actual reason when nothing succeeded (e.g. missing ffmpeg)
        # instead of a bare "0/N" that reads like the format is unsupported.
        if done == 0 and last_error is not None:
            _notify("whisper-daemon", "Transcription failed", last_error[:200])
        else:
            _notify(
                "whisper-daemon",
                "Transcription complete",
                f"{done}/{len(files)} files transcribed.",
            )
        self._reset_meeting_ui()


def _make_item(title: str, action: str | None, target: object) -> NSMenuItem:
    """Create an NSMenuItem with the given title, action selector, and target."""
    item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
        title, action, ""
    )
    if target is not None:
        item.setTarget_(target)
    return item


def _notify(title: str, subtitle: str, message: str) -> None:
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


def _save_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    """Save float32 audio array as 16-bit WAV file."""
    int16_audio = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16_audio.tobytes())
    logger.info("Audio saved: %s (%.1fs)", path, len(audio) / sample_rate)


def _collect_futures(
    futures: dict,
    all_results: list[tuple[float, dict]],
    wait: bool = False,
) -> bool:
    """Collect completed transcription futures. Returns True if any were collected."""
    if wait and futures:
        done_set, _ = concurrent.futures.wait(futures.keys())
    else:
        done_set = {f for f in futures if f.done()}

    collected = False
    for future in done_set:
        start_time = futures.pop(future)
        try:
            result = future.result()
            if result.get("text", "").strip():
                all_results.append((start_time, result))
                collected = True
        except Exception as exc:
            logger.error("Chunk transcription failed: %s", exc)
    return collected


def _write_partial(path: Path, results: list[tuple[float, dict]]) -> None:
    """Write current transcript-so-far to a live file."""
    sorted_results = sorted(results, key=lambda r: r[0])
    text = " ".join(r.get("text", "").strip() for _, r in sorted_results if r.get("text", "").strip())
    try:
        path.write_text(text, encoding="utf-8")
    except Exception:
        pass


def _start_exit_watchdog(timeout: float, exit_code: int = 0) -> None:
    """Force-exit the process if it's still alive `timeout` seconds from now.

    Backstop for a shutdown that wedges in websockets/MLX/resource_tracker —
    we have logs showing the process lingering after "Shutdown requested"
    with no "Daemon stopped" line, so the user has to kill -9 it. This
    guarantees exit within `timeout` seconds regardless.

    ``exit_code`` 0 means "user asked us to quit" (launchd will NOT
    restart); non-zero means "we died wedged" (launchd restarts us).
    Callers stopping a meeting must pass a timeout that accommodates the
    meeting save (minutes), not just socket teardown.
    """
    def _kill() -> None:
        time.sleep(timeout)
        logger.warning(
            "Exit watchdog fired after %.0fs — forcing os._exit(%d)",
            timeout, exit_code,
        )
        os._exit(exit_code)

    threading.Thread(target=_kill, name="exit-watchdog", daemon=True).start()


def run_with_menubar(
    daemon: object,
    hotkey_listener: object,
    on_appkit_ready: Callable[[], None] | None = None,
    on_delegate_ready: Callable[["MenuBarDelegate"], None] | None = None,
) -> None:
    """Run the daemon event loop in a background thread, menu bar on main thread.

    The main thread MUST run the NSApplication event loop for AppKit to work.
    The daemon event loop runs in a daemon thread.

    ``on_appkit_ready`` is called on the main thread after NSApplication is
    configured but before the event loop starts — use it for APIs that
    require the AppKit run loop (e.g. NSEvent global monitors).

    ``on_delegate_ready`` receives the MenuBarDelegate so callers (e.g. signal
    handlers) can trigger a graceful meeting stop before exit.
    """
    daemon_thread = threading.Thread(target=daemon.run, daemon=True)
    daemon_thread.start()

    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)

    delegate = MenuBarDelegate.alloc().initWithDaemon_hotkeyListener_(
        daemon, hotkey_listener
    )
    app.setDelegate_(delegate)

    if on_delegate_ready is not None:
        on_delegate_ready(delegate)

    if on_appkit_ready is not None:
        on_appkit_ready()

    AppHelper.runEventLoop()
