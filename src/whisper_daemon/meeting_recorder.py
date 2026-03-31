"""Long-form meeting recorder with VAD-based chunk splitting."""

import logging
import queue
import time
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

from whisper_daemon.vad import SileroVAD

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "float32"
BLOCK_SIZE = 512
VAD_THRESHOLD = 0.5
DEFAULT_CHUNK_SILENCE = 2.0
MAX_CHUNK_SEC = 120  # Force split after 2min to keep chunks manageable


@dataclass(frozen=True)
class AudioChunk:
    """A chunk of audio with its position in the recording timeline."""

    audio: np.ndarray
    start_time: float  # seconds from recording start
    duration: float  # seconds


class MeetingRecorder:
    """Records audio continuously and splits into chunks at natural pauses.

    Chunks are emitted to a queue for parallel transcription while
    recording continues.
    """

    def __init__(
        self,
        chunk_queue: queue.Queue[AudioChunk | None],
        device: str | int | None = None,
        chunk_silence: float = DEFAULT_CHUNK_SILENCE,
    ) -> None:
        self._chunk_queue = chunk_queue
        self._device = device
        self._chunk_silence = chunk_silence

        self._vad = SileroVAD()

        self._stream: sd.InputStream | None = None
        self._frames: list[np.ndarray] = []
        self._recording = False

        self._recording_start: float = 0.0
        self._chunk_start: float = 0.0
        self._voice_detected_in_chunk = False
        self._silence_start: float | None = None

    def start(self) -> None:
        """Start continuous recording."""
        self._frames = []
        self._recording = True
        self._recording_start = time.monotonic()
        self._chunk_start = 0.0
        self._voice_detected_in_chunk = False
        self._silence_start = None
        self._vad.reset_states()

        self._stream = sd.InputStream(
            device=self._device,
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=BLOCK_SIZE,
            callback=self._callback,
        )
        self._stream.start()

        device_name = self._device or "system default"
        logger.info("Meeting recording started (device: %s)", device_name)

    def stop(self) -> None:
        """Stop recording and emit any remaining audio as a final chunk."""
        self._recording = False

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        self._emit_chunk()
        self._chunk_queue.put(None)  # sentinel: no more chunks
        self._vad.reset_states()

        elapsed = time.monotonic() - self._recording_start
        logger.info("Meeting recording stopped — total %.1fs", elapsed)

    def _callback(
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

        self._frames.append(indata.copy())

        chunk_elapsed = self._current_chunk_duration()
        if chunk_elapsed >= MAX_CHUNK_SEC:
            logger.info("Chunk max duration reached (%.0fs), splitting", MAX_CHUNK_SEC)
            self._emit_chunk()
            return

        speech_prob = self._vad(indata[:, 0].copy())
        now = time.monotonic()

        if speech_prob > VAD_THRESHOLD:
            self._voice_detected_in_chunk = True
            self._silence_start = None
        elif self._voice_detected_in_chunk:
            if self._silence_start is None:
                self._silence_start = now
            elif now - self._silence_start >= self._chunk_silence:
                logger.info("Pause detected (%.1fs silence), splitting chunk", self._chunk_silence)
                self._emit_chunk()

    def _emit_chunk(self) -> None:
        """Concatenate accumulated frames and put chunk on queue."""
        if not self._frames:
            return

        audio = np.concatenate(self._frames, axis=0).squeeze()
        duration = len(audio) / SAMPLE_RATE

        if duration < 0.3:
            self._frames = []
            return

        chunk = AudioChunk(
            audio=audio,
            start_time=self._chunk_start,
            duration=duration,
        )
        self._chunk_queue.put(chunk)
        logger.info(
            "Chunk emitted: start=%.1fs, duration=%.1fs",
            chunk.start_time, chunk.duration,
        )

        self._chunk_start += duration
        self._frames = []
        self._voice_detected_in_chunk = False
        self._silence_start = None
        self._vad.reset_states()

    def _current_chunk_duration(self) -> float:
        total_samples = sum(f.shape[0] for f in self._frames)
        return total_samples / SAMPLE_RATE
