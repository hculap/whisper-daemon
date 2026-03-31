"""Menu bar status icon with interactive actions using rumps."""

import concurrent.futures
import logging
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import rumps
from AppKit import NSOpenPanel
from AppKit import NSFileHandlingPanelOKButton

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

TITLES = {
    State.IDLE: "Ready",
    State.RECORDING: "Recording...",
    State.TRANSCRIBING: "Transcribing...",
}


class MenuBarApp(rumps.App):
    """Menu bar icon with status display and interactive actions."""

    def __init__(self, daemon: object, hotkey_listener: object) -> None:
        super().__init__(ICONS[State.IDLE], quit_button=None)
        self._daemon = daemon
        self._hotkey = hotkey_listener

        self._meeting_active = False
        self._meeting_start: float = 0.0
        self._meeting_thread: threading.Thread | None = None

        self._status_item = rumps.MenuItem("Status: Ready", callback=None)
        self._meeting_item = rumps.MenuItem(
            "Start Meeting Recording", callback=self._toggle_meeting
        )
        self._transcribe_files_item = rumps.MenuItem(
            "Transcribe Files...", callback=self._transcribe_files
        )
        self._transcribe_folder_item = rumps.MenuItem(
            "Transcribe Folder...", callback=self._transcribe_folder
        )

        self.menu = [
            self._status_item,
            None,
            self._meeting_item,
            None,
            self._transcribe_files_item,
            self._transcribe_folder_item,
            None,
            rumps.MenuItem("Quit", callback=self._quit),
        ]

        self._poll_timer = rumps.Timer(self._poll_state, 0.3)
        self._poll_timer.start()
        self._last_state = State.IDLE

    def _poll_state(self, _timer: rumps.Timer) -> None:
        if self._meeting_active:
            elapsed = time.monotonic() - self._meeting_start
            mins, secs = divmod(int(elapsed), 60)
            self.title = "🔴"
            self._meeting_item.title = f"Stop Recording ({mins}:{secs:02d})"
            self._status_item.title = f"Meeting recording ({mins}:{secs:02d})"
            return

        state = self._daemon._state
        if state != self._last_state:
            self._last_state = state
            self.title = ICONS.get(state, "🎙")
            self._status_item.title = f"Status: {TITLES.get(state, 'Unknown')}"

    def _toggle_meeting(self, _sender: rumps.MenuItem) -> None:
        if self._meeting_active:
            self._stop_meeting()
        else:
            self._start_meeting()

    def _start_meeting(self) -> None:
        self._meeting_active = True
        self._meeting_start = time.monotonic()
        self._meeting_item.title = "Stop Recording (0:00)"
        self.title = "🔴"
        logger.info("Meeting recording started from menu bar")

        self._meeting_thread = threading.Thread(
            target=self._meeting_worker, daemon=True
        )
        self._meeting_thread.start()

    def _stop_meeting(self) -> None:
        self._meeting_active = False
        self._meeting_item.title = "Start Meeting Recording"
        self.title = "⏳"
        self._status_item.title = "Finishing transcription..."
        logger.info("Meeting recording stop requested from menu bar")

    def _meeting_worker(self) -> None:
        from whisper_daemon.formats import to_txt
        from whisper_daemon.meeting_recorder import AudioChunk, MeetingRecorder
        from whisper_daemon.transcriber import transcribe as transcribe_audio

        model = self._daemon._model
        chunk_queue: queue.Queue[AudioChunk | None] = queue.Queue()
        recorder = MeetingRecorder(chunk_queue)

        all_results: list[tuple[float, str]] = []
        chunk_count = 0

        recorder.start()

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                futures: dict[concurrent.futures.Future, float] = {}

                while self._meeting_active:
                    try:
                        chunk = chunk_queue.get(timeout=0.5)
                    except queue.Empty:
                        self._collect_meeting_results(futures, all_results)
                        continue

                    if chunk is None:
                        break

                    chunk_count += 1
                    logger.info("Meeting chunk %d: %.1fs", chunk_count, chunk.duration)
                    future = pool.submit(transcribe_audio, chunk.audio, model)
                    futures[future] = chunk.start_time
                    self._collect_meeting_results(futures, all_results)

                recorder.stop()

                while True:
                    try:
                        chunk = chunk_queue.get_nowait()
                    except queue.Empty:
                        break
                    if chunk is None:
                        break
                    chunk_count += 1
                    text = transcribe_audio(chunk.audio, model)
                    if text:
                        all_results.append((chunk.start_time, text))

                self._collect_meeting_results(futures, all_results, wait=True)

        except Exception:
            logger.exception("Meeting recording failed")
            rumps.notification(
                "whisper-daemon", "Error", "Meeting recording failed. Check logs."
            )
            self._reset_meeting_ui()
            return

        if not all_results:
            rumps.notification("whisper-daemon", "Done", "No speech detected.")
            self._reset_meeting_ui()
            return

        all_results.sort(key=lambda r: r[0])
        full_text = " ".join(text for _, text in all_results if text)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path.home() / "Desktop" / f"meeting_{timestamp}.txt"
        output_path.write_text(full_text, encoding="utf-8")

        logger.info("Meeting transcript saved to %s", output_path)
        rumps.notification(
            "whisper-daemon",
            f"Meeting recorded ({chunk_count} chunks)",
            str(output_path),
        )
        self._reset_meeting_ui()

    def _collect_meeting_results(
        self,
        futures: dict,
        all_results: list[tuple[float, str]],
        wait: bool = False,
    ) -> None:
        if wait and futures:
            done_futures, _ = concurrent.futures.wait(futures.keys())
        else:
            done_futures = {f for f in futures if f.done()}

        for future in done_futures:
            start_time = futures.pop(future)
            try:
                text = future.result()
                if text:
                    all_results.append((start_time, text))
            except Exception as exc:
                logger.error("Meeting chunk failed: %s", exc)

    def _reset_meeting_ui(self) -> None:
        self._meeting_active = False
        self._meeting_item.title = "Start Meeting Recording"
        self.title = ICONS[State.IDLE]
        self._status_item.title = "Status: Ready"

    def _transcribe_files(self, _sender: rumps.MenuItem) -> None:
        panel = NSOpenPanel.openPanel()
        panel.setCanChooseFiles_(True)
        panel.setCanChooseDirectories_(False)
        panel.setAllowsMultipleSelection_(True)
        panel.setAllowedFileTypes_(AUDIO_VIDEO_EXTENSIONS)
        panel.setTitle_("Select audio/video files to transcribe")

        if panel.runModal() != NSFileHandlingPanelOKButton:
            return

        paths = [str(url.path()) for url in panel.URLs()]
        if paths:
            threading.Thread(
                target=self._transcribe_paths_worker, args=(paths,), daemon=True
            ).start()

    def _transcribe_folder(self, _sender: rumps.MenuItem) -> None:
        panel = NSOpenPanel.openPanel()
        panel.setCanChooseFiles_(False)
        panel.setCanChooseDirectories_(True)
        panel.setAllowsMultipleSelection_(False)
        panel.setTitle_("Select folder to transcribe")

        if panel.runModal() != NSFileHandlingPanelOKButton:
            return

        folder = str(panel.URLs()[0].path())
        if folder:
            threading.Thread(
                target=self._transcribe_paths_worker, args=([folder],), daemon=True
            ).start()

    def _transcribe_paths_worker(self, paths: list[str]) -> None:
        from whisper_daemon.formats import to_txt
        from whisper_daemon.transcriber import transcribe_file

        model = self._daemon._model

        self.title = "⏳"
        self._status_item.title = "Transcribing files..."

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
            rumps.notification("whisper-daemon", "No files", "No audio/video files found.")
            self._reset_meeting_ui()
            return

        done = 0
        for file_path in files:
            try:
                self._status_item.title = f"Transcribing {file_path.name}..."
                result = transcribe_file(str(file_path), model=model)
                text = result.get("text", "").strip()
                if text:
                    output = file_path.with_suffix(".txt")
                    output.write_text(text, encoding="utf-8")
                    done += 1
            except Exception:
                logger.exception("Failed to transcribe %s", file_path)

        rumps.notification(
            "whisper-daemon",
            f"Transcription complete",
            f"{done}/{len(files)} files transcribed.",
        )
        self._reset_meeting_ui()

    def _quit(self, _sender: rumps.MenuItem) -> None:
        logger.info("Quit from menu bar")
        if self._meeting_active:
            self._stop_meeting()
        self._hotkey.stop()
        self._daemon.shutdown()
        rumps.quit_application()


def run_with_menubar(daemon: object, hotkey_listener: object) -> None:
    """Run the daemon event loop in a background thread, menu bar on main thread."""
    daemon_thread = threading.Thread(target=daemon.run, daemon=True)
    daemon_thread.start()

    app = MenuBarApp(daemon, hotkey_listener)
    app.run()
