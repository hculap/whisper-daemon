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
from Foundation import NSRunLoop, NSDefaultRunLoopMode
from PyObjCTools import AppHelper

from whisper_daemon.config import VALID_FORMATS, Settings, load_settings, save_settings
from whisper_daemon.daemon import State

logger = logging.getLogger(__name__)

AUDIO_VIDEO_EXTENSIONS = [
    "mp3", "wav", "m4a", "flac", "ogg", "webm",
    "mp4", "mkv", "avi", "mov", "aac", "wma",
]

ICONS = {
    State.IDLE: "\U0001f399",       # 🎙
    State.RECORDING: "\U0001f534",  # 🔴
    State.TRANSCRIBING: "\u231b",   # ⏳
}

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
        self._last_state = State.IDLE
        self._settings = load_settings()

        return self

    def applicationDidFinishLaunching_(self, notification):
        self._setup_status_bar()
        self._start_poll_timer()

    def _setup_status_bar(self):
        status_bar = NSStatusBar.systemStatusBar()
        self._status_item = status_bar.statusItemWithLength_(
            NSVariableStatusItemLength
        )
        self._status_item.setTitle_(ICONS[State.IDLE])
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
            self._timer, NSDefaultRunLoopMode
        )

    # -- Poll timer --

    @objc.typedSelector(b"v@:@")
    def pollState_(self, timer):
        if self._meeting_active:
            elapsed = time.monotonic() - self._meeting_start
            mins, secs = divmod(int(elapsed), 60)
            self._status_item.setTitle_("\U0001f534")
            self._meeting_menu_item.setTitle_(f"Stop Recording ({mins}:{secs:02d})")
            self._status_menu_item.setTitle_(f"Meeting recording ({mins}:{secs:02d})")
            return

        state = self._daemon._state
        if state != self._last_state:
            self._last_state = state
            self._status_item.setTitle_(ICONS.get(state, "\U0001f399"))
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
        self._hotkey.stop()
        self._daemon.shutdown()
        AppHelper.stopEventLoop()

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
    def onToggleScreenshots_(self, sender):
        self._settings.capture_screenshots = not self._settings.capture_screenshots
        sender.setState_(1 if self._settings.capture_screenshots else 0)
        save_settings(self._settings)
        logger.info("Capture screenshots: %s", self._settings.capture_screenshots)

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

    # -- Meeting recording --

    def _start_meeting(self):
        self._meeting_active = True
        self._meeting_start = time.monotonic()
        self._meeting_menu_item.setTitle_("Stop Recording (0:00)")
        self._status_item.setTitle_("\U0001f534")
        logger.info("Meeting recording started from menu bar")

        self._meeting_thread = threading.Thread(
            target=self._meeting_worker, daemon=True
        )
        self._meeting_thread.start()

    def _stop_meeting(self):
        self._meeting_active = False
        self._meeting_menu_item.setTitle_("Start Meeting Recording")
        self._status_item.setTitle_("\u231b")
        self._status_menu_item.setTitle_("Finishing transcription...")
        logger.info("Meeting recording stop requested from menu bar")

    def _meeting_worker(self):
        from whisper_daemon.meeting_recorder import AudioChunk, MeetingRecorder
        from whisper_daemon.screen_capture import ScreenCapture
        from whisper_daemon.transcriber import transcribe_full

        model = self._daemon._model
        device = self._settings.recording_device or None
        chunk_queue: queue.Queue[AudioChunk | None] = queue.Queue()
        recorder = MeetingRecorder(chunk_queue, device=device)

        all_results: list[tuple[float, dict]] = []
        all_audio: list[np.ndarray] = []
        chunk_count = 0

        # Prepare output dir early for screenshots
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rec_dir = self._settings.recording_dir_path / f"recording_{timestamp}"
        rec_dir.mkdir(parents=True, exist_ok=True)

        screen_capture: ScreenCapture | None = None
        if self._settings.capture_screenshots:
            screen_capture = ScreenCapture(
                rec_dir, interval=self._settings.screenshot_interval
            )
            screen_capture.start()

        recorder.start()

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                futures: dict[concurrent.futures.Future, float] = {}

                while self._meeting_active:
                    try:
                        chunk = chunk_queue.get(timeout=0.5)
                    except queue.Empty:
                        _collect_futures(futures, all_results)
                        continue

                    if chunk is None:
                        break

                    chunk_count += 1
                    logger.info(
                        "Meeting chunk %d: %.1fs", chunk_count, chunk.duration
                    )
                    if self._settings.save_audio:
                        all_audio.append(chunk.audio.copy())
                    future = pool.submit(transcribe_full, chunk.audio, model)
                    futures[future] = chunk.start_time
                    _collect_futures(futures, all_results)

                recorder.stop()
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
                        break
                    chunk_count += 1
                    if self._settings.save_audio:
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

        all_results.sort(key=lambda r: r[0])

        # Merge results with timestamp-adjusted segments
        merged_segments: list[dict] = []
        merged_text_parts: list[str] = []
        for start_offset, result in all_results:
            text = result.get("text", "").strip()
            if text:
                merged_text_parts.append(text)
            for seg in result.get("segments", []):
                merged_segments.append({
                    **seg,
                    "start": seg["start"] + start_offset,
                    "end": seg["end"] + start_offset,
                })
        merged_result = {
            "text": " ".join(merged_text_parts),
            "segments": merged_segments,
            "language": all_results[0][1].get("language", "") if all_results else "",
        }

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

        logger.info("Meeting saved: %s", ", ".join(written))
        _notify(
            "whisper-daemon",
            f"Meeting recorded ({chunk_count} chunks{screenshots_msg})",
            str(rec_dir),
        )
        self._reset_meeting_ui()

    def _reset_meeting_ui(self):
        self._meeting_active = False
        self._meeting_menu_item.setTitle_("Start Meeting Recording")
        self._status_item.setTitle_(ICONS[State.IDLE])
        self._status_menu_item.setTitle_("Status: Ready")

    # -- File transcription --

    def _transcribe_paths_worker(self, paths: list[str]) -> None:
        from whisper_daemon.formats import FORMATTERS
        from whisper_daemon.transcriber import transcribe_file

        model = self._daemon._model

        self._status_item.setTitle_("\u231b")
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
) -> None:
    """Collect completed transcription futures."""
    if wait and futures:
        done_set, _ = concurrent.futures.wait(futures.keys())
    else:
        done_set = {f for f in futures if f.done()}

    for future in done_set:
        start_time = futures.pop(future)
        try:
            result = future.result()
            if result.get("text", "").strip():
                all_results.append((start_time, result))
        except Exception as exc:
            logger.error("Chunk transcription failed: %s", exc)


def run_with_menubar(daemon: object, hotkey_listener: object) -> None:
    """Run the daemon event loop in a background thread, menu bar on main thread.

    The main thread MUST run the NSApplication event loop for AppKit to work.
    The daemon event loop runs in a daemon thread.
    """
    daemon_thread = threading.Thread(target=daemon.run, daemon=True)
    daemon_thread.start()

    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)

    delegate = MenuBarDelegate.alloc().initWithDaemon_hotkeyListener_(
        daemon, hotkey_listener
    )
    app.setDelegate_(delegate)

    AppHelper.runEventLoop()
