"""Menu bar status icon using raw pyobjc (NSStatusBar + NSMenu).

Replaces rumps which is broken on macOS 14+/Sequoia — menus don't drop down,
multiple phantom icons appear, and @clicked decorators silently fail.
"""

import concurrent.futures
import logging
import queue
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
        self._last_state = State.IDLE
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
        )
        self._browser_recorder = None  # set during meeting with browser source

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
        logger.info("Quit from menu bar")
        if self._meeting_active:
            self._stop_meeting()
        self._browser_bridge.stop()
        self._hotkey.stop()
        self._daemon.shutdown()
        AppHelper.stopEventLoop()

    def _set_icon(self, state: State) -> None:
        """Set the menu bar icon emoji for the given state."""
        self._status_item.button().setTitle_(ICONS.get(state, "🎙"))

    def _set_icon_by_name(self, icon_text: str, _fallback: str = "") -> None:
        """Set the menu bar icon to the given text."""
        self._status_item.button().setTitle_(icon_text)

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
        self._settings.capture_screenshots = not self._settings.capture_screenshots
        sender.setState_(1 if self._settings.capture_screenshots else 0)
        save_settings(self._settings)
        logger.info("Capture screenshots: %s", self._settings.capture_screenshots)

    @objc.typedSelector(b"v@:@")
    def onToggleDiarize_(self, sender):
        self._settings.diarize = not self._settings.diarize
        sender.setState_(1 if self._settings.diarize else 0)
        save_settings(self._settings)
        logger.info("Speaker diarization: %s", self._settings.diarize)

    @objc.typedSelector(b"v@:@")
    def onToggleAutoRecord_(self, sender):
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
        self._settings.recording_dir = path
        save_settings(self._settings)
        self._rec_dir_item.setTitle_(
            self._format_dir_label("Recording Folder", path)
        )

    @objc.typedSelector(b"v@:@")
    def onToggleRecFmt_(self, sender):
        fmt = str(sender.title())
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
        self._settings.transcription_output_dir = path
        save_settings(self._settings)
        self._trans_dir_item.setTitle_(
            self._format_dir_label("Transcription Output", path)
        )

    @objc.typedSelector(b"v@:@")
    def onToggleTransFmt_(self, sender):
        fmt = str(sender.title())
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

        self._settings.recording_device = device_name
        save_settings(self._settings)
        logger.info("Recording device changed to: %s", device_name or "system default")

    @staticmethod
    def _format_dir_label(prefix: str, path: str) -> str:
        display = path.replace(str(Path.home()), "~") if path else "same as input"
        return f"{prefix}: {display}"

    # -- Browser bridge callbacks (called from asyncio thread) --

    def _on_browser_connect(self, title: str, url: str) -> None:
        """Extension started a capture session."""
        logger.info("Browser meeting started: %s (%s)", title, url)
        if self._settings.auto_record_meetings and not self._meeting_active:
            AppHelper.callAfter(self._start_meeting, True, title)

    def _on_browser_audio(self, data: bytes) -> None:
        """Raw PCM audio from browser tab."""
        if self._browser_recorder is not None:
            samples = np.frombuffer(data, dtype=np.float32)
            self._browser_recorder.feed_audio(samples)

    def _on_browser_disconnect(self) -> None:
        """Extension disconnected or stopped capture."""
        logger.info("Browser meeting ended")
        if self._meeting_active and self._meeting_browser_triggered:
            AppHelper.callAfter(self._stop_meeting)

    # -- Meeting recording --

    def _start_meeting(self, browser_triggered: bool = False, browser_title: str = "") -> None:
        self._meeting_active = True
        self._meeting_browser_triggered = browser_triggered
        self._meeting_start = time.monotonic()
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

        # Mic recorder (local device)
        mic_recorder = MeetingRecorder(chunk_queue, device=device, source_label="mic")

        # Browser recorder (Chrome extension tab audio)
        browser_recorder: MeetingRecorder | None = None
        if browser_triggered and self._browser_bridge.connected:
            browser_recorder = MeetingRecorder(chunk_queue, source_label="browser")
            browser_recorder.start_without_device()
            self._browser_recorder = browser_recorder

        # Number of recorders determines how many None sentinels to expect
        sentinel_expected = 1 + (1 if browser_recorder else 0)
        sentinel_count = 0

        all_results: list[tuple[float, dict]] = []
        all_audio: list[np.ndarray] = []
        chunk_count = 0

        # Prepare output dir early for screenshots
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        title_slug = browser_title.replace(" ", "_")[:30] if browser_title else ""
        dir_name = f"recording_{title_slug}_{timestamp}" if title_slug else f"recording_{timestamp}"
        rec_dir = self._settings.recording_dir_path / dir_name
        rec_dir.mkdir(parents=True, exist_ok=True)

        screen_capture: ScreenCapture | None = None
        if self._settings.capture_screenshots:
            screen_capture = ScreenCapture(
                rec_dir, interval=self._settings.screenshot_interval
            )
            screen_capture.start()

        HEALTH_CHECK_SEC = 120.0  # first warning after 2 min of silence
        HEALTH_REPEAT_SEC = 120.0  # re-warn every 2 min while still silent
        RECOVERY_BACKOFF_SEC = 10.0  # min interval between recovery attempts

        telemetry.meeting_start()
        try:
            mic_recorder.start()
        except Exception as exc:
            logger.error("Failed to open mic for meeting: %s", exc)
            if screen_capture:
                screen_capture.stop()
            self._meeting_active = False
            self._meeting_menu_item.setTitle_("Start Meeting Recording")
            return

        if mic_recorder.fell_back_to_default:
            _notify(
                "whisper-daemon",
                "Microphone fallback",
                "Preferred device unavailable — using system default mic.",
            )

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                futures: dict[concurrent.futures.Future, float] = {}

                partial_path = rec_dir / "transcript_live.txt"
                last_chunk_time = time.monotonic()
                last_health_warn = 0.0
                last_recovery_attempt = 0.0
                mic_lost_notified = False

                while self._meeting_active:
                    now = time.monotonic()

                    # Check if mic recorder needs device recovery. Retries
                    # keep firing on a backoff timer until they succeed or
                    # the meeting ends — devices (Bluetooth, USB) can
                    # reappear minutes later, so we never give up.
                    if mic_recorder.needs_recovery and now - last_recovery_attempt >= RECOVERY_BACKOFF_SEC:
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
                        # Health check: warn repeatedly while silent, and
                        # ask the mic recorder to try reopening its stream
                        # — the callback path can't detect "alive but
                        # delivering silence" reliably, so the loop itself
                        # has to nudge it when nothing is arriving.
                        elapsed_silent = now - last_chunk_time
                        if (
                            elapsed_silent > HEALTH_CHECK_SEC
                            and now - last_health_warn >= HEALTH_REPEAT_SEC
                        ):
                            last_health_warn = now
                            logger.warning(
                                "No audio chunks for %.0fs — mic may not be capturing speech",
                                elapsed_silent,
                            )
                            _notify(
                                "whisper-daemon",
                                "No speech detected",
                                f"No audio for {int(elapsed_silent)}s. Check your microphone.",
                            )
                            mic_recorder.request_recovery()
                        continue

                    if chunk is None:
                        sentinel_count += 1
                        if sentinel_count >= sentinel_expected:
                            break
                        continue

                    chunk_count += 1
                    last_chunk_time = now
                    logger.info(
                        "Meeting chunk %d: %.1fs [%s]",
                        chunk_count, chunk.duration, chunk.source,
                    )
                    telemetry.meeting_chunk_queued(chunk_count, chunk.duration, chunk.start_time)
                    if self._settings.save_audio or self._settings.diarize:
                        all_audio.append(chunk.audio.copy())
                    cn = chunk_count  # capture for closure
                    def _transcribe_and_track(audio, m, n):
                        result = transcribe_full(audio, m)
                        segs = len(result.get("segments", []))
                        chars = len(result.get("text", ""))
                        telemetry.meeting_chunk_transcribed(n, chars, segs)
                        return result
                    future = pool.submit(_transcribe_and_track, chunk.audio, model, cn)
                    futures[future] = chunk.start_time
                    if _collect_futures(futures, all_results):
                        _write_partial(partial_path, all_results)
                        self._send_results_to_browser(all_results, chunk_count)

                mic_recorder.stop()
                if browser_recorder is not None:
                    browser_recorder.stop_without_device()
                    self._browser_recorder = None
                if screen_capture is not None:
                    screen_capture.stop()

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
                    if self._settings.save_audio or self._settings.diarize:
                        all_audio.append(chunk.audio.copy())
                    result = transcribe_full(chunk.audio, model)
                    if result.get("text", "").strip():
                        all_results.append((chunk.start_time, result))

        except Exception:
            logger.exception("Meeting recording failed")
            _notify("whisper-daemon", "Error", "Meeting recording failed.")
            self._reset_meeting_ui()
            return

        if not all_results:
            _notify("whisper-daemon", "Done", "No speech detected.")
            self._reset_meeting_ui()
            return

        from whisper_daemon.formats import merge_chunk_results
        merged_result = merge_chunk_results(all_results)

        if self._settings.diarize and all_audio:
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
        for fmt in self._settings.recording_formats:
            if fmt in FORMATTERS:
                out = rec_dir / f"transcript.{fmt}"
                out.write_text(FORMATTERS[fmt](merged_result), encoding="utf-8")
                written.append(str(out))

        if self._settings.save_audio and all_audio:
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
        self._reset_meeting_ui()

    def _send_results_to_browser(self, all_results: list, chunk_count: int) -> None:
        """Forward transcription results to the Chrome extension if connected."""
        if not self._browser_bridge.connected:
            return
        for start_time, result in all_results[-3:]:  # send recent results
            text = result.get("text", "").strip()
            if text:
                self._browser_bridge.send_chunk_result(text, start_time, chunk_count)

    def graceful_stop(self, timeout: float = 300.0) -> None:
        """Stop any active meeting, wait for save/diarize, then quit.

        Called from signal handlers to ensure the meeting is properly saved
        before the process exits. Runs on the signal-handler thread — must
        not touch AppKit.
        """
        if self._meeting_active:
            logger.info("Graceful stop: stopping active meeting before exit")
            self._meeting_active = False
            if self._meeting_thread is not None and self._meeting_thread.is_alive():
                self._meeting_thread.join(timeout=timeout)
                if self._meeting_thread.is_alive():
                    logger.warning("Meeting worker did not finish within %.0fs", timeout)
        self._browser_bridge.stop()

    def _reset_meeting_ui(self) -> None:
        self._meeting_active = False
        self._meeting_browser_triggered = False
        self._browser_recorder = None
        self._meeting_menu_item.setTitle_("Start Meeting Recording")
        self._set_icon(State.IDLE)
        self._status_menu_item.setTitle_("Status: Ready")

    # -- File transcription --

    def _transcribe_paths_worker(self, paths: list[str]) -> None:
        from whisper_daemon.formats import FORMATTERS
        from whisper_daemon.transcriber import transcribe_file

        model = self._daemon._model

        self._set_icon(State.TRANSCRIBING)
        self._status_menu_item.setTitle_("Transcribing files...")

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
        for file_path in files:
            try:
                self._status_menu_item.setTitle_(
                    f"Transcribing {file_path.name}..."
                )
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
            except Exception:
                logger.exception("Failed to transcribe %s", file_path)

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
