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
from whisper_daemon.notify import notify as _notify

logger = logging.getLogger(__name__)

AUDIO_VIDEO_EXTENSIONS = [
    "mp3", "wav", "m4a", "flac", "ogg", "webm",
    "mp4", "mkv", "avi", "mov", "aac", "wma",
]

ICONS = {
    State.IDLE: "🎙",
    State.RECORDING: "🔴",
    State.TRANSCRIBING: "⏳",
    State.REVIEW: "✏️",
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
    State.REVIEW: "Review dictation — accept or cancel",
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
        self._meeting_store = None
        self._review_server = None
        self._last_meeting_dir = None
        self._meeting_paused = False
        self._diarization_status = {"available": False, "message": "Speaker recognition not checked"}
        self._browser_audio_lock = threading.RLock()
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
        self._last_history = ()
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
        self._browser_prebuffer: list[tuple[bytes, int]] = []
        self._meeting_session_id = None
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

        self._preview_item = _make_item("Dictation preview: —", None, None)
        self._preview_item.setEnabled_(False)
        menu.addItem_(self._preview_item)
        menu.addItem_(_make_item("Review / Edit Dictation…", "onReviewDictation:", self))
        menu.addItem_(_make_item("Cancel Dictation", "onCancelDictation:", self))
        dictation_menu = NSMenu.alloc().init()
        self._confirm_item = _make_item("Review before pasting", "onConfirmDictation:", self)
        self._confirm_item.setState_(1 if self._settings.dictation_confirm else 0)
        dictation_menu.addItem_(self._confirm_item)
        for index, language in enumerate(("Auto", "Polski", "English")):
            item = _make_item(f"Language: {language}", "onDictationLanguage:", self)
            item.setTag_(index)
            dictation_menu.addItem_(item)
        dictation_menu.addItem_(_make_item("Silence before stop…", "onDictationSilence:", self))
        dictation_menu.addItem_(_make_item("Vocabulary…", "onDictationVocabulary:", self))
        for index, action in enumerate(("Record", "Paste last", "Cancel", "Accept", "Speak")):
            item = _make_item(f"Shortcut: {action}…", "onDictationShortcut:", self)
            item.setTag_(index)
            dictation_menu.addItem_(item)
        item = _make_item("Dictation Settings", None, None)
        item.setSubmenu_(dictation_menu)
        menu.addItem_(item)

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
        menu.addItem_(_make_item("Pause / Resume Meeting", "onPauseMeeting:", self))
        menu.addItem_(_make_item("Review Last Meeting…", "onReviewMeeting:", self))
        menu.addItem_(_make_item("Open Meeting…", "onOpenMeeting:", self))

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
            "Continuous Audio Saved (required for recovery)", None, None
        )
        self._save_audio_item.setState_(1)
        self._save_audio_item.setEnabled_(False)
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
            "Start meetings explicitly in Chrome", None, None
        )
        self._auto_record_item.setEnabled_(False)
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

        # Speaker diarization for "Transcribe Files/Folder" — separate from the
        # meeting-recording toggle above so a slow pyannote pass can be enabled
        # for imported files only (or vice versa).
        self._trans_diarize_item = _make_item(
            "Transcription Diarization", "onToggleTransDiarize:", self
        )
        if self._settings.transcription_diarize:
            self._trans_diarize_item.setState_(1)
        settings_menu.addItem_(self._trans_diarize_item)

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
        preview = self._daemon._pending_text.replace("\n", " ")
        self._preview_item.setTitle_("Preview: " + (preview[-180:] or "—"))
        if self._meeting_active:
            elapsed = time.monotonic() - self._meeting_start
            mins, secs = divmod(int(elapsed), 60)
            self._set_icon(State.RECORDING)
            self._meeting_menu_item.setTitle_(f"Stop Recording ({mins}:{secs:02d})")
            self._status_menu_item.setTitle_(f"Meeting {'paused' if self._meeting_paused else 'recording'} ({mins}:{secs:02d})")
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
        meeting_active = self._meeting_phase != "idle" or bool(self._meeting_thread and self._meeting_thread.is_alive())
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
        if tuple(history) == self._last_history: return
        self._last_history = tuple(history)
        self._recent_menu.removeAllItems()
        for index, text in enumerate(history):
            item = _make_item(text[:70].replace("\n", " "), "onCopyRecent:", self)
            item.setTag_(index)
            item.setToolTip_("Paste this dictation into the focused input")
            self._recent_menu.addItem_(item)

    def onCopyRecent_(self, sender):
        from whisper_daemon.events import Event, EventType
        from whisper_daemon.focus import capture_target
        history = self._daemon.history
        if 0 <= sender.tag() < len(history):
            self._daemon._queue.put(Event(EventType.PASTE_HISTORY, (history[sender.tag()], capture_target())))

    def onCancelDictation_(self, sender):
        from whisper_daemon.events import Event, EventType
        self._daemon._queue.put(Event(EventType.RECORD_CANCEL))

    def onReviewDictation_(self, sender):
        from whisper_daemon.events import Event, EventType
        if self._daemon._state != State.REVIEW:
            _notify("whisper-daemon", "Dictation review", "Enable Review before pasting in Dictation Settings for editable drafts.")
            return
        generation = self._daemon._recording_gen
        shortcut = self._settings.accept_hotkey
        text = self._ask_text("Review dictation", self._daemon._review_text,
            f"Save updates the draft. Return to the original text field and press {shortcut} to paste it.")
        if text is not None:
            self._daemon._queue.put(Event(EventType.DICTATION_EDIT, {'generation':generation, 'text':text}))
            _notify("whisper-daemon", "Draft updated", f"Return to the original field and press {shortcut} to paste.")

    def _ask_text(self, title, value, description):
        from AppKit import NSAlert, NSTextField
        alert = NSAlert.alloc().init()
        alert.setMessageText_(title)
        alert.setInformativeText_(description)
        field = NSTextField.alloc().initWithFrame_(((0, 0), (480, 80)))
        field.setStringValue_(value)
        field.setEditable_(True)
        field.setSelectable_(True)
        alert.setAccessoryView_(field)
        alert.addButtonWithTitle_("Save")
        alert.addButtonWithTitle_("Cancel")
        return str(field.stringValue()) if alert.runModal() == 1000 else None

    def _save_dictation_settings(self):
        self._daemon._settings = self._settings
        save_settings(self._settings)

    def onConfirmDictation_(self, sender):
        self._settings.dictation_confirm = not self._settings.dictation_confirm
        sender.setState_(1 if self._settings.dictation_confirm else 0)
        self._save_dictation_settings()

    def onDictationLanguage_(self, sender):
        self._settings.dictation_language = ("auto", "pl", "en")[sender.tag()]
        self._save_dictation_settings()
        _notify("whisper-daemon", "Dictation language", self._settings.dictation_language)

    def onDictationSilence_(self, sender):
        value = self._ask_text("Silence before stop", str(self._settings.dictation_silence), "Seconds, from 0 to 30. Use 0 to stop only with the shortcut.")
        if value is None: return
        try:
            seconds = float(value)
            if not 0 <= seconds <= 30: raise ValueError()
            self._settings.dictation_silence = seconds
            self._save_dictation_settings()
        except ValueError:
            _notify("whisper-daemon", "Invalid silence duration", "Enter a number between 0 and 30.")

    def onDictationVocabulary_(self, sender):
        value = self._ask_text("Dictation vocabulary", ", ".join(self._settings.dictation_vocabulary), "Names and specialist words, separated by commas. Used as transcription hints.")
        if value is not None:
            self._settings.dictation_vocabulary = [word.strip()[:100] for word in value.split(",") if word.strip()][:100]
            self._save_dictation_settings()

    def onDictationShortcut_(self, sender):
        import dataclasses
        from whisper_daemon.shortcuts import parse_shortcut
        action = ("record", "paste", "cancel", "accept", "speak")[sender.tag()]
        key = action + "_hotkey"
        value = self._ask_text("Shortcut: " + action, getattr(self._settings, key), "Example: cmd+shift+space. Modifiers: cmd, shift, alt, ctrl.")
        if value is None: return
        try:
            parse_shortcut(value)
            changed = dataclasses.replace(self._settings, **{key: value.lower()})
            self._hotkey.configure(changed)
            self._settings = changed
            self._save_dictation_settings()
        except ValueError as exc:
            _notify("whisper-daemon", "Invalid shortcut", str(exc))

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

    @objc.typedSelector(b"v@:@")
    def onToggleTransDiarize_(self, sender):
        with self._settings_lock:
            self._settings.transcription_diarize = (
                not self._settings.transcription_diarize
            )
        sender.setState_(1 if self._settings.transcription_diarize else 0)
        save_settings(self._settings)
        logger.info(
            "Transcription diarization: %s", self._settings.transcription_diarize
        )

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

    def _on_browser_connect(self, title: str, url: str, session_id: str, started_at: float) -> None:
        """Extension started a capture session.

        A {"type":"start"} from the extension is always an explicit user
        gesture now (it follows the getDisplayMedia permission click), so it
        must always start the meeting when none is active — regardless of the
        auto_record_meetings preference, which only governs fully-automatic
        starts and is intentionally not consulted here.
        """
        if self._meeting_phase != "idle" or (self._meeting_thread and self._meeting_thread.is_alive()): return False
        with self._browser_audio_lock:
            self._browser_prebuffer = []
            self._browser_prebuffer_bytes = 0
        logger.info("Browser capture requested")
        def start_if_current():
            # Stop may arrive before Cocoa runs this callback. A cancelled
            # session must never open the microphone later.
            if self._browser_bridge.session_id == session_id:
                self._start_meeting(True, title, started_at, session_id)
        AppHelper.callAfter(start_if_current)

    def _on_browser_audio(self, data: bytes, position: int) -> None:
        """Stage the startup tail, then hand off atomically to its source."""
        with self._browser_audio_lock:
            if self._browser_recorder is not None:
                self._browser_recorder.feed_audio(np.frombuffer(data, dtype="<f4"), position=position)
                return
            if self._meeting_phase == "transcribing": return
            if self._browser_prebuffer_bytes + len(data) > BROWSER_PREBUFFER_MAX_BYTES:
                self._browser_bridge.send_json_str('{"type":"error","message":"Audio startup buffer full; stop and retry"}')
                AppHelper.callAfter(self._stop_meeting)
                return
            self._browser_prebuffer.append((data, position))
            self._browser_prebuffer_bytes += len(data)

    def _on_browser_disconnect(self, session_id: str) -> None:
        """Extension disconnected or stopped capture."""
        logger.info("Browser meeting ended")
        def stop_if_browser():
            if self._meeting_session_id == session_id and self._meeting_phase != 'idle':
                if self._meeting_active and self._meeting_browser_triggered:
                    self._stop_meeting()
                # Its worker owns ACKed staging, including during preflight.
                return
            if self._browser_bridge.session_id is None:
                with self._browser_audio_lock:
                    self._browser_prebuffer = []
                    self._browser_prebuffer_bytes = 0
        AppHelper.callAfter(stop_if_browser)

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

    def _run_off_bridge(self, fn, *args) -> None:
        """Run blocking work off the bridge event-loop thread.

        Device enumeration (sounddevice) and disk I/O must never run inline on
        the WebSocket coroutine thread — while they block, the async-for loop
        can't read the next audio frame, silently starving a live meeting.
        """
        threading.Thread(target=fn, args=args, daemon=True).start()

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
                build_status_message(phase, elapsed, self._meeting_chunk_count, snapshot=True)
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

        if msg_type == "audio_gap":
            _notify("whisper-daemon", "Participant audio gap", f'{data["samples"] / 16000:.2f} seconds missing; silence inserted and marked.')

        elif msg_type == "get_settings":
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

        elif msg_type == "pause_meeting":
            AppHelper.callAfter(self.onPauseMeeting_, None)

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
            self._save_audio_item.setState_(1)
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

    def _start_meeting(self, browser_triggered: bool = False, browser_title: str = "", started_at=None, session_id=None) -> None:
        # Guard against duplicate starts: two rapid browser 'start' messages
        # each schedule _start_meeting via callAfter, and _on_browser_connect
        # checks _meeting_active before the first one has run — so the check
        # there is not enough. A second concurrent meeting would open a
        # second mic stream and corrupt the chunk pipeline.
        if self._meeting_phase != "idle" or (self._meeting_thread is not None and self._meeting_thread.is_alive()):
            logger.info("Meeting already active — ignoring duplicate start")
            return
        self._meeting_active = True
        self._meeting_browser_triggered = browser_triggered
        self._meeting_start = time.monotonic() if started_at is None else started_at
        self._meeting_stopped_at = None
        self._meeting_session_id = session_id
        self._meeting_paused = False
        self._meeting_pause_events = []
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
        self._meeting_stopped_at = time.monotonic()
        self._browser_bridge.end_session()
        self._meeting_phase = "transcribing"
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
        from whisper_daemon.meeting_recorder import MeetingRecorder
        from whisper_daemon.recording_store import RecordingStore, DiskChunkQueue, atomic_json
        from whisper_daemon.formats import FORMATTERS, merge_chunk_results
        from whisper_daemon.screen_capture import ScreenCapture
        from whisper_daemon.transcriber import transcribe_full
        import uuid

        settings = self._settings
        slug = re.sub(r"[^A-Za-z0-9_-]", "", browser_title.replace(" ", "_"))[:30]
        rec_dir = settings.recording_dir_path / f"recording_{slug}_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
        store = None
        recorders = []
        screens = None
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        pending = None
        stopped = False
        failure = None
        language = None if settings.recording_language == "auto" else settings.recording_language
        try:
            diarize_ready = False
            if settings.diarize:
                from whisper_daemon.model_policy import diarization_status
                self._diarization_status = diarization_status()
                diarize_ready = self._diarization_status['available']
                if not diarize_ready:
                    _notify("whisper-daemon", "Speaker recognition unavailable", self._diarization_status['message'])
                self._browser_bridge.send_json_str(__import__('json').dumps({"type": "capabilities", "diarization": self._diarization_status}))
            sources = (['mic'] if settings.capture_mic else []) + (['browser'] if settings.capture_tab and browser_triggered else [])
            store = RecordingStore(rec_dir, started_at=self._meeting_start, sources=sources)
            self._attach_recording_store(store)
            jobs = DiskChunkQueue(rec_dir)
            if settings.capture_mic and self._meeting_active:
                mic = MeetingRecorder(jobs, device=settings.recording_device or None,
                    raw_sink=lambda audio: store.append("mic", audio),
                    paused=lambda: self._meeting_paused)
                try:
                    mic.start()
                    recorders.append(mic)
                except Exception as exc:
                    mic.abort()
                    store.meta['errors'].append(f"Microphone: {exc}")
                    _notify("whisper-daemon", "Microphone unavailable", str(exc)[:180])
            if settings.capture_tab and browser_triggered:
                browser = MeetingRecorder(jobs, source_label="browser",
                    raw_sink=lambda audio, position: store.append("browser", audio, at=position),
                    sample_filter=store.unpaused_audio)
                browser.start_without_device()
                recorders.append(browser)
                # The same lock protects staging and handoff: no PCM can land
                # between draining the prebuffer and publishing the recorder.
                with self._browser_audio_lock:
                    for data, position in self._browser_prebuffer:
                        browser.feed_audio(np.frombuffer(data, dtype="<f4"), position=position)
                    self._browser_prebuffer = []
                    self._browser_prebuffer_bytes = 0
                    self._browser_recorder = browser
            if not recorders:
                raise RuntimeError("No audio source is available")
            if settings.capture_screenshots and self._meeting_active:
                screens = ScreenCapture(rec_dir, interval=settings.screenshot_interval,
                                        displays=settings.screenshot_displays, paused=lambda: self._meeting_paused)
                screens.start()
            last_status = 0.0
            last_recovery = 0.0
            while True:
                now = time.monotonic()
                if not self._meeting_active and not stopped:
                    self._meeting_phase = "transcribing"
                    with self._browser_audio_lock:
                        self._browser_recorder = None
                        for recorder in recorders:
                            if recorder._source_label == "mic": recorder.stop()
                            else: recorder.stop_without_device()
                    if screens: screens.stop()
                    store.finish(duration=(self._meeting_stopped_at or now) - self._meeting_start, state="transcribing")
                    stopped = True
                for recorder in recorders:
                    if recorder.error:
                        raise RuntimeError(f"Audio storage failed: {recorder.error}")
                    if not stopped and recorder.check_liveness(now):
                        store.meta['errors'].append('Microphone stopped delivering audio; recovery requested')
                        _notify("whisper-daemon", "Microphone interrupted", "Trying to reconnect. Any missing audio will be marked in the recording.")
                    if (not stopped and recorder._source_label == "mic"
                            and recorder.needs_recovery and now - last_recovery > 10):
                        last_recovery = now
                        recorder.recover_device()
                if pending is not None and pending[0].done():
                    future, chunk = pending
                    result = future.result()  # failures keep the durable job for recovery
                    jobs.ack(chunk, result)
                    pending = None
                    self._meeting_chunk_count += 1
                    results = jobs.results()
                    _write_partial(rec_dir / "transcript_live.txt", results)
                    self._send_results_to_browser(results, self._meeting_chunk_count)
                if pending is None:
                    try:
                        chunk = jobs.get_nowait()
                    except queue.Empty:
                        if stopped: break
                    else:
                        pending = (pool.submit(transcribe_full, chunk.audio, self._daemon._model, language), chunk)
                if now - last_status >= 1:
                    last_status = now
                    self._browser_bridge.send_json_str(build_status_message(
                        self._meeting_phase, now - self._meeting_start, self._meeting_chunk_count,
                        session_id=self._meeting_session_id))
                    self._browser_bridge.send_json_str(__import__("json").dumps({
                        "type": "health", "sources": store.health(), "paused": self._meeting_paused}))
                time.sleep(.05)
            merged = merge_chunk_results(jobs.results())
            merged['duration'] = store.meta['duration']
            if diarize_ready:
                try:
                    from whisper_daemon.diarizer import diarize_file
                    from whisper_daemon.diarize_merge import merge_speakers_with_transcript
                    merged = merge_speakers_with_transcript(diarize_file(str(rec_dir / "recording.wav")), merged)
                except Exception as exc:
                    store.meta['errors'].append(f"Speaker recognition unavailable: {exc}")
                    _notify("whisper-daemon", "Speaker recognition unavailable", "Audio and transcript saved without automatic speaker labels.")
            atomic_json(rec_dir / "transcript.json", merged)
            for fmt in settings.recording_formats:
                if fmt in FORMATTERS:
                    target = rec_dir / f"transcript.{fmt}"
                    temp = target.with_suffix(target.suffix + ".tmp")
                    temp.write_text(FORMATTERS[fmt](merged), encoding="utf-8")
                    temp.replace(target)
            from whisper_daemon.meeting_review import prepare_review
            prepare_review(rec_dir)
            self._last_meeting_dir = rec_dir
            store.meta['state'] = 'complete'
            store._save()
            (rec_dir / "transcript_live.txt").unlink(missing_ok=True)
            _notify("whisper-daemon", "Meeting saved", str(rec_dir))
        except Exception as exc:
            failure = exc
            logger.exception("Meeting failed; source audio and pending jobs remain recoverable")
            _notify("whisper-daemon", "Meeting interrupted", f"{exc}. Recoverable files: {rec_dir}")
        finally:
            with self._browser_audio_lock:
                self._browser_recorder = None
                for recorder in recorders:
                    recorder.abort()
            if screens: screens.stop()
            if store and not stopped:
                try:
                    if failure: store.meta['errors'].append(str(failure))
                    store.finish(duration=time.monotonic() - self._meeting_start, state="interrupted")
                except Exception:
                    logger.exception("Export failed; raw source files retained")
            pool.shutdown(wait=False)
            self._meeting_store = None
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
        self._browser_bridge.end_session()
        self._meeting_phase = "idle"
        self._browser_recorder = None
        self._browser_prebuffer = []
        self._browser_prebuffer_bytes = 0

        # Tell the extension the meeting is over so its in-bar button goes idle.
        try:
            self._browser_bridge.send_json_str(
                build_status_message("idle", 0.0, 0, session_id=self._meeting_session_id)
            )
        except Exception:
            logger.debug("Failed to push idle status", exc_info=True)

        def _update_ui() -> None:
            self._meeting_menu_item.setTitle_("Start Meeting Recording")
            self._set_icon(State.IDLE)
            self._status_menu_item.setTitle_("Status: Ready")

        AppHelper.callAfter(_update_ui)

    # -- File transcription --

    def _finish_file_ui(self) -> None:
        if self._meeting_phase == "idle":
            self._set_icon_safe(self._daemon._state)
            self._set_status_safe("Ready")

    def _open_review(self, directory):
        import webbrowser
        from whisper_daemon.meeting_review import ReviewServer
        if self._review_server:
            self._review_server.shutdown()
            self._review_server.server_close()
        self._review_server = ReviewServer(directory)
        webbrowser.open(self._review_server.start())

    def onReviewMeeting_(self, sender):
        directory = self._last_meeting_dir
        if directory is None:
            candidates = list(self._settings.recording_dir_path.glob("recording_*/transcript.json"))
            if candidates: directory = max(candidates, key=lambda p: p.stat().st_mtime).parent
        if directory: self._run_off_bridge(self._open_review, directory)
        else: _notify("whisper-daemon", "No completed meetings", "Use Open Meeting to select a recording folder.")

    def onOpenMeeting_(self, sender):
        from AppKit import NSOpenPanel
        panel = NSOpenPanel.openPanel()
        panel.setCanChooseFiles_(False)
        panel.setCanChooseDirectories_(True)
        panel.setAllowsMultipleSelection_(False)
        if panel.runModal() == 1:
            directory = Path(str(panel.URLs()[0].path()))
            if (directory / "transcript.json").exists(): self._run_off_bridge(self._open_review, directory)
            else: _notify("whisper-daemon", "No transcript", "Select a meeting folder containing transcript.json.")

    def onPauseMeeting_(self, sender) -> None:
        if not self._meeting_active: return
        with self._browser_audio_lock:
            self._meeting_paused = not self._meeting_paused
            elapsed = time.monotonic() - self._meeting_start
            self._meeting_pause_events.append((elapsed, self._meeting_paused))
            if self._meeting_store: self._meeting_store.pause(self._meeting_paused, at=elapsed)
        self._set_status_safe("Meeting paused" if self._meeting_paused else "Recording meeting")

    def _attach_recording_store(self, store):
        with self._browser_audio_lock:
            for elapsed, paused in self._meeting_pause_events:
                store.pause(paused, at=elapsed)
            self._meeting_store = store

    def _transcribe_paths_worker(self, paths: list[str]) -> None:
        from whisper_daemon.formats import FORMATTERS
        from whisper_daemon.transcriber import transcribe_file

        model = self._daemon._model
        diarize = self._settings.transcription_diarize

        if self._meeting_phase == "idle":
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
            self._finish_file_ui()
            return

        out_dir = self._settings.transcription_output_dir_path
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)

        done = 0
        last_error: str | None = None
        diarize_error: str | None = None
        for file_path in files:
            try:
                self._set_status_safe(f"Transcribing {file_path.name}...")
                result = transcribe_file(str(file_path), model=model)

                if diarize:
                    # A diarization failure (missing HF token, model download,
                    # unsupported audio) must not cost the user the transcript —
                    # fall through and write it without speaker labels.
                    self._set_status_safe(f"Diarizing {file_path.name}...")
                    try:
                        result = _diarize_transcript(str(file_path), result)
                    except Exception as exc:
                        diarize_error = str(exc) or exc.__class__.__name__
                        logger.exception("Diarization failed for %s", file_path)

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
        elif diarize_error is not None:
            _notify(
                "whisper-daemon",
                "Transcription complete — diarization failed",
                f"{done}/{len(files)} files transcribed. {diarize_error[:150]}",
            )
        else:
            _notify(
                "whisper-daemon",
                "Transcription complete",
                f"{done}/{len(files)} files transcribed.",
            )
        self._finish_file_ui()


def _diarize_transcript(path: str, result: dict) -> dict:
    """Label a transcript's segments with speakers (pyannote; needs an HF token).

    Imported at call time — pyannote/torch load is seconds-slow and must not be
    paid by users who never turn diarization on.
    """
    from whisper_daemon.diarize_merge import merge_speakers_with_transcript
    from whisper_daemon.diarizer import diarize_file

    speaker_segments = diarize_file(path)
    if not speaker_segments:
        logger.warning("No speaker segments for %s — writing plain transcript", path)
        return result

    merged = merge_speakers_with_transcript(speaker_segments, result)
    logger.info(
        "Diarized %s — %d speaker(s)", path, len(merged.get("speakers", []))
    )
    return merged


def _make_item(title: str, action: str | None, target: object) -> NSMenuItem:
    """Create an NSMenuItem with the given title, action selector, and target."""
    item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
        title, action, ""
    )
    if target is not None:
        item.setTarget_(target)
    return item


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
