"""Audio recording with Silero VAD for automatic silence detection."""

import logging
import queue
import time

import numpy as np
import sounddevice as sd

from whisper_daemon.events import Event, EventType
from whisper_daemon.vad import SileroVAD

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "float32"
BLOCK_SIZE = 512  # Matches Silero VAD expected chunk size (32ms at 16kHz)
MAX_RECORDING_SEC = 120
VAD_THRESHOLD = 0.5
DEFAULT_SILENCE_SEC = 1.5


class AudioRecorder:
    """Records audio from the default mic with Silero VAD auto-stop."""

    def __init__(
        self,
        event_queue: queue.Queue[Event],
        silence_timeout: float = DEFAULT_SILENCE_SEC,
    ) -> None:
        self._queue = event_queue
        self._silence_timeout = silence_timeout

        self._vad = SileroVAD()

        self._stream: sd.InputStream | None = None
        self._chunks: list[np.ndarray] = []
        self._voice_detected = False
        self._silence_start: float | None = None
        self._recording_start: float = 0.0
        self._recording = False

    def start_recording(self) -> None:
        """Start capturing audio from the default microphone."""
        self._chunks = []
        self._voice_detected = False
        self._silence_start = None
        self._recording_start = time.monotonic()
        self._recording = True
        self._vad.reset_states()

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=BLOCK_SIZE,
            callback=self._audio_callback,
        )
        self._stream.start()
        logger.info("Recording started")

    def stop_recording(self) -> np.ndarray:
        """Stop recording and return the captured audio as a 1D numpy array."""
        self._recording = False

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        self._vad.reset_states()

        if not self._chunks:
            return np.array([], dtype=np.float32)

        audio = np.concatenate(self._chunks, axis=0).squeeze()
        self._chunks = []
        logger.info("Recording stopped — %.1fs of audio", len(audio) / SAMPLE_RATE)
        return audio

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            logger.warning("Audio status: %s", status)

        if not self._recording:
            return

        self._chunks.append(indata.copy())

        elapsed = time.monotonic() - self._recording_start
        if elapsed > MAX_RECORDING_SEC:
            logger.warning("Max recording time reached (%ds)", MAX_RECORDING_SEC)
            self._queue.put(Event(EventType.RECORD_STOP))
            return

        speech_prob = self._vad(indata[:, 0].copy())

        now = time.monotonic()

        if speech_prob > VAD_THRESHOLD:
            self._voice_detected = True
            self._silence_start = None
        elif self._voice_detected:
            if self._silence_start is None:
                self._silence_start = now
            elif now - self._silence_start >= self._silence_timeout:
                logger.info("VAD: silence detected (%.1fs)", self._silence_timeout)
                self._queue.put(Event(EventType.RECORD_STOP))
