"""Daemon state machine and event loop."""

import logging
import queue
import threading
import time
from enum import Enum, auto

from whisper_daemon.events import Event, EventType
from whisper_daemon.paster import paste_text
from whisper_daemon.recorder import AudioRecorder
from whisper_daemon.sounds import play_error, play_start, play_stop
from whisper_daemon.transcriber import transcribe

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
    ) -> None:
        self._queue = event_queue
        self._recorder = recorder
        self._model = model
        self._state = State.IDLE
        self._running = False
        self._recording_started_at: float = 0.0

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
                continue

            self._handle_event(event)

        self._cleanup()
        logger.info("Daemon stopped")

    def _handle_event(self, event: Event) -> None:
        if event.type == EventType.RECORD_TOGGLE:
            self._handle_toggle()
        elif event.type == EventType.RECORD_STOP:
            self._handle_record_stop()
        elif event.type == EventType.TRANSCRIPTION_DONE:
            self._handle_transcription_done(event.payload)
        elif event.type == EventType.ERROR:
            self._handle_error(event.payload)

    def _handle_toggle(self) -> None:
        if self._state == State.IDLE:
            self._state = State.RECORDING
            self._recording_started_at = time.monotonic()
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

    def _start_transcription(self) -> None:
        self._state = State.TRANSCRIBING
        audio = self._recorder.stop_recording()
        play_stop()
        logger.info("State: RECORDING -> TRANSCRIBING")

        if audio.size == 0:
            logger.warning("No audio captured")
            self._queue.put(Event(EventType.TRANSCRIPTION_DONE, ""))
            return

        thread = threading.Thread(
            target=self._transcribe_worker,
            args=(audio,),
            daemon=True,
        )
        thread.start()

    def _transcribe_worker(self, audio: object) -> None:
        try:
            text = transcribe(audio, model=self._model)
            self._queue.put(Event(EventType.TRANSCRIPTION_DONE, text))
        except Exception as exc:
            logger.exception("Transcription worker failed")
            self._queue.put(Event(EventType.ERROR, str(exc)))

    def _handle_transcription_done(self, text: str) -> None:
        if text:
            paste_text(text)
        else:
            logger.info("Empty transcription, nothing to paste")

        self._state = State.IDLE
        logger.info("State: TRANSCRIBING -> IDLE")

    def _handle_error(self, message: str) -> None:
        play_error()
        logger.error("Error: %s", message)
        if self._state == State.RECORDING:
            self._recorder.stop_recording()
        self._state = State.IDLE
        logger.info("State: -> IDLE (error recovery)")

    def _cleanup(self) -> None:
        if self._state == State.RECORDING:
            self._recorder.stop_recording()
            logger.info("Stopped in-progress recording during cleanup")
