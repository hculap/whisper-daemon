"""Daemon state machine and event loop."""

import logging
import queue
import threading
import time
from enum import Enum, auto

import numpy as np

from whisper_daemon.events import Event, EventType
from whisper_daemon.paster import paste_text
from whisper_daemon.recorder import AudioRecorder
from whisper_daemon.sounds import play_error, play_start, play_stop
from whisper_daemon.transcriber import transcribe

logger = logging.getLogger(__name__)

GPU_WARMUP_INTERVAL = 60.0  # seconds between idle GPU warm-ups


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
    ) -> None:
        self._queue = event_queue
        self._recorder = recorder
        self._model = model
        self._state = State.IDLE
        self._running = False
        self._recording_started_at: float = 0.0
        self._history: list[str] = []
        self._max_history = 5

        # Progressive transcription state
        self._pending_text: str = ""
        self._pending_samples: int = 0
        self._preview_thread: threading.Thread | None = None

        # GPU warm-up
        self._last_transcription_time: float = time.monotonic()

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
        logger.info("Daemon running (state=IDLE)")

        while self._running:
            try:
                event = self._queue.get(timeout=0.5)
            except queue.Empty:
                self._maybe_warmup_gpu()
                continue

            self._handle_event(event)

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
        elif event.type == EventType.ERROR:
            self._handle_error(event.payload)

    def _handle_toggle(self) -> None:
        if self._state == State.IDLE:
            self._state = State.RECORDING
            self._recording_started_at = time.monotonic()
            self._pending_text = ""
            self._pending_samples = 0
            self._recorder.start_recording()
            play_start()
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
            self._last_transcription_time = time.monotonic()
            logger.info("Preview done: %d chars", len(text))
        except Exception:
            logger.exception("Preview transcription failed")

    def _start_transcription(self) -> None:
        self._state = State.TRANSCRIBING
        audio = self._recorder.stop_recording()
        play_stop()
        logger.info("State: RECORDING -> TRANSCRIBING")

        if audio.size == 0:
            logger.warning("No audio captured")
            self._queue.put(Event(EventType.TRANSCRIPTION_DONE, ""))
            return

        # If we have a recent preview and audio hasn't grown much since, use it
        new_samples = len(audio) - self._pending_samples
        new_seconds = new_samples / 16000
        if self._pending_text and new_seconds < 1.5:
            logger.info(
                "Using preview result (%.1fs new audio since preview, skipping re-transcription)",
                new_seconds,
            )
            self._queue.put(Event(EventType.TRANSCRIPTION_DONE, self._pending_text))
            self._last_transcription_time = time.monotonic()
            return

        # Full transcription needed
        thread = threading.Thread(
            target=self._transcribe_worker,
            args=(audio,),
            daemon=True,
        )
        thread.start()

    def _transcribe_worker(self, audio: np.ndarray) -> None:
        try:
            text = transcribe(audio, model=self._model)
            self._last_transcription_time = time.monotonic()
            self._queue.put(Event(EventType.TRANSCRIPTION_DONE, text))
        except Exception as exc:
            logger.exception("Transcription worker failed")
            self._queue.put(Event(EventType.ERROR, str(exc)))

    def _handle_transcription_done(self, text: str) -> None:
        if text:
            paste_text(text)
            self._history.insert(0, text)
            if len(self._history) > self._max_history:
                self._history.pop()
        else:
            logger.info("Empty transcription, nothing to paste")

        self._pending_text = ""
        self._pending_samples = 0
        self._state = State.IDLE
        logger.info("State: TRANSCRIBING -> IDLE")

    def _handle_paste_last(self) -> None:
        if self._history:
            paste_text(self._history[0])
            logger.info("Pasted last transcription (%d chars)", len(self._history[0]))
        else:
            logger.info("No transcription history to paste")

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
        """Keep Metal GPU warm with periodic dummy inference."""
        if self._state != State.IDLE:
            return
        elapsed = time.monotonic() - self._last_transcription_time
        if elapsed < GPU_WARMUP_INTERVAL:
            return
        self._last_transcription_time = time.monotonic()
        threading.Thread(target=self._gpu_warmup, daemon=True).start()

    def _gpu_warmup(self) -> None:
        try:
            transcribe(np.zeros(8000, dtype=np.float32), model=self._model)
            logger.debug("GPU warm-up done")
        except Exception:
            pass

    def _cleanup(self) -> None:
        if self._state == State.RECORDING:
            self._recorder.stop_recording()
            logger.info("Stopped in-progress recording during cleanup")
