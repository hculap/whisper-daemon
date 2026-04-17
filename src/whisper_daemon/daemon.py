"""Daemon state machine and event loop."""

import logging
import queue
import threading
import time
from enum import Enum, auto

import numpy as np

from whisper_daemon import telemetry
from whisper_daemon.events import Event, EventType
from whisper_daemon.paster import paste_text
from whisper_daemon.recorder import AudioRecorder
from whisper_daemon.sounds import play_error, play_start, play_stop
from whisper_daemon.transcriber import transcribe
from whisper_daemon.tts import speak_text

logger = logging.getLogger(__name__)


class State(Enum):
    IDLE = auto()
    RECORDING = auto()
    TRANSCRIBING = auto()


class Daemon:
    """Main daemon: consumes events from the queue and drives state transitions."""

    def __init__(
        self,
        event_queue: queue.Queue[Event],
        recorder: AudioRecorder,
        model: str = "mlx-community/whisper-large-v3-turbo-q4",
        settings: object | None = None,
    ) -> None:
        self._queue = event_queue
        self._recorder = recorder
        self._model = model
        self._settings = settings
        self._state = State.IDLE
        self._running = False
        self._recording_started_at: float = 0.0
        self._history: list[str] = []
        self._max_history = 5

        # Progressive transcription state
        self._pending_text: str = ""
        self._pending_samples: int = 0
        self._preview_thread: threading.Thread | None = None


    @property
    def history(self) -> list[str]:
        return list(self._history)

    @property
    def running(self) -> bool:
        return self._running

    def shutdown(self) -> None:
        """Signal the event loop to stop."""
        logger.info("Shutdown requested")
        self._running = False

    def run(self) -> None:
        """Main event loop — blocks until shutdown() is called."""
        self._running = True
        telemetry.start_session()
        logger.info("Daemon running (state=IDLE)")

        while self._running:
            try:
                event = self._queue.get(timeout=0.5)
            except queue.Empty:
                self._maybe_warmup_gpu()
                continue

            try:
                self._handle_event(event)
            except Exception:
                logger.exception("Unhandled error in event loop")
                play_error()
                self._state = State.IDLE

        self._cleanup()
        logger.info("Daemon stopped")

    def _handle_event(self, event: Event) -> None:
        if event.type == EventType.RECORD_TOGGLE:
            self._handle_toggle()
        elif event.type == EventType.RECORD_STOP:
            self._handle_record_stop()
        elif event.type == EventType.TRANSCRIBE_PREVIEW:
            self._handle_preview()
        elif event.type == EventType.TRANSCRIPTION_DONE:
            self._handle_transcription_done(event.payload)
        elif event.type == EventType.PASTE_LAST:
            self._handle_paste_last()
        elif event.type == EventType.SPEAK_CLIPBOARD:
            self._handle_speak_clipboard()
        elif event.type == EventType.ERROR:
            self._handle_error(event.payload)

    def _handle_toggle(self) -> None:
        if self._state == State.IDLE:
            try:
                self._recorder.start_recording()
            except Exception as exc:
                logger.error("Failed to open audio device: %s", exc)
                play_error()
                return
            self._state = State.RECORDING
            self._recording_started_at = time.monotonic()
            self._pending_text = ""
            self._pending_samples = 0
            play_start()
            telemetry.mark("record_start")
            logger.info("State: IDLE -> RECORDING")
        elif self._state == State.RECORDING:
            elapsed = time.monotonic() - self._recording_started_at
            if elapsed < 0.5:
                logger.info("Ignoring toggle — too soon (%.2fs), debounce", elapsed)
                return
            self._start_transcription()
        elif self._state == State.TRANSCRIBING:
            logger.info("Ignoring hotkey during transcription")

    def _handle_record_stop(self) -> None:
        if self._state == State.RECORDING:
            self._start_transcription()

    def _handle_preview(self) -> None:
        """Snapshot audio buffer and transcribe in background while recording continues."""
        if self._state != State.RECORDING:
            return

        # Don't start a new preview if previous is still running
        if self._preview_thread is not None and self._preview_thread.is_alive():
            return

        audio, total_samples = self._recorder.get_audio_snapshot()
        if audio.size == 0:
            return

        telemetry.mark("preview_start", audio_sec=round(len(audio) / 16000, 1))
        logger.info("Preview: transcribing %.1fs snapshot", len(audio) / 16000)
        self._preview_thread = threading.Thread(
            target=self._preview_worker,
            args=(audio, total_samples),
            daemon=True,
        )
        self._preview_thread.start()

    def _preview_worker(self, audio: np.ndarray, total_samples: int) -> None:
        """Transcribe a snapshot of the audio buffer."""
        try:
            text = transcribe(audio, model=self._model)
            self._pending_text = text
            self._pending_samples = total_samples
            telemetry.mark("preview_done", chars=len(text))
            logger.info("Preview done: %d chars", len(text))
        except Exception:
            logger.exception("Preview transcription failed")

    def _start_transcription(self) -> None:
        self._state = State.TRANSCRIBING
        audio = self._recorder.stop_recording()
        voice_detected = self._recorder.voice_detected
        play_stop()

        # Wait for any in-flight preview to finish (avoid concurrent GPU access).
        # No timeout — concurrent MLX/Metal calls cause SIGABRT.
        if self._preview_thread is not None and self._preview_thread.is_alive():
            logger.info("Waiting for preview to finish before final transcription...")
            self._preview_thread.join()

        telemetry.mark("record_stop", audio_sec=round(len(audio) / 16000, 1) if audio.size > 0 else 0)
        logger.info("State: RECORDING -> TRANSCRIBING")

        if audio.size == 0:
            logger.warning("No audio captured")
            self._queue.put(Event(EventType.TRANSCRIPTION_DONE, ""))
            return

        # Skip transcription when VAD never caught voice — on silent input Whisper
        # hallucinates short English phrases ("Thanks for watching.", etc) and we'd
        # paste that garbage into the focused app.
        if not voice_detected:
            logger.warning(
                "Discarding %.1fs of audio — no voice detected (check microphone)",
                len(audio) / 16000,
            )
            play_error()
            self._queue.put(Event(EventType.TRANSCRIPTION_DONE, ""))
            return

        # If we have a recent preview and audio hasn't grown much since, use it
        new_samples = len(audio) - self._pending_samples
        new_seconds = new_samples / 16000
        if self._pending_text and new_seconds < 2.5:
            telemetry.mark("transcribe_start", mode="preview_cached", new_audio_sec=round(new_seconds, 1))
            telemetry.mark("transcribe_done", chars=len(self._pending_text))
            logger.info(
                "Using preview result (%.1fs new audio since preview, skipping re-transcription)",
                new_seconds,
            )
            self._queue.put(Event(EventType.TRANSCRIPTION_DONE, self._pending_text))
            return

        # Full transcription needed
        telemetry.mark("transcribe_start", mode="full", new_audio_sec=round(new_seconds, 1))
        thread = threading.Thread(
            target=self._transcribe_worker,
            args=(audio,),
            daemon=True,
        )
        thread.start()

    def _transcribe_worker(self, audio: np.ndarray) -> None:
        try:
            text = transcribe(audio, model=self._model)
            telemetry.mark("transcribe_done", chars=len(text))
            self._queue.put(Event(EventType.TRANSCRIPTION_DONE, text))
        except Exception as exc:
            logger.exception("Transcription worker failed")
            self._queue.put(Event(EventType.ERROR, str(exc)))

    def _handle_transcription_done(self, text: str) -> None:
        if text:
            paste_text(text)
            telemetry.mark("paste_done", chars=len(text))
            self._history.insert(0, text)
            if len(self._history) > self._max_history:
                self._history.pop()
        else:
            logger.info("Empty transcription, nothing to paste")

        self._pending_text = ""
        self._pending_samples = 0
        self._state = State.IDLE
        telemetry.flush()
        logger.info("State: TRANSCRIBING -> IDLE")

    def _handle_paste_last(self) -> None:
        if self._history:
            paste_text(self._history[0])
            logger.info("Pasted last transcription (%d chars)", len(self._history[0]))
        else:
            logger.info("No transcription history to paste")

    def _handle_speak_clipboard(self) -> None:
        """Simulate Cmd+C to copy selected text, then speak it via TTS."""
        from AppKit import NSPasteboard, NSPasteboardTypeString
        from Quartz import (
            CGEventCreateKeyboardEvent,
            CGEventPost,
            CGEventSetFlags,
            kCGEventFlagMaskCommand,
            kCGHIDEventTap,
        )

        C_KEYCODE = 8

        event_down = CGEventCreateKeyboardEvent(None, C_KEYCODE, True)
        CGEventSetFlags(event_down, kCGEventFlagMaskCommand)
        CGEventPost(kCGHIDEventTap, event_down)

        event_up = CGEventCreateKeyboardEvent(None, C_KEYCODE, False)
        CGEventSetFlags(event_up, kCGEventFlagMaskCommand)
        CGEventPost(kCGHIDEventTap, event_up)

        time.sleep(0.15)

        pb = NSPasteboard.generalPasteboard()
        text = pb.stringForType_(NSPasteboardTypeString)

        if text:
            tts_lang = "auto"
            if self._settings is not None:
                tts_lang = getattr(self._settings, "tts_language", "auto")
            logger.info("Speak clipboard: %d chars (lang=%s)", len(text), tts_lang)
            speak_text(str(text), language=tts_lang)
        else:
            logger.info("Clipboard empty, nothing to speak")

    def _handle_error(self, message: str) -> None:
        play_error()
        logger.error("Error: %s", message)
        if self._state == State.RECORDING:
            self._recorder.stop_recording()
        self._pending_text = ""
        self._pending_samples = 0
        self._state = State.IDLE
        logger.info("State: -> IDLE (error recovery)")

    def _maybe_warmup_gpu(self) -> None:
        """No-op. GPU warmup removed — it held the GIL for ~2s every 60s,
        freezing the AppKit main thread and making the menu bar unresponsive.
        The model stays loaded in memory; cold GPU penalty is ~1-2s on first
        transcription after idle, which is acceptable.
        """

    def _cleanup(self) -> None:
        if self._state == State.RECORDING:
            self._recorder.stop_recording()
            logger.info("Stopped in-progress recording during cleanup")
