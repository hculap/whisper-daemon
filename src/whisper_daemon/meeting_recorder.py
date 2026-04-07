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
RMS_SILENCE_FLOOR = 0.005  # Below this RMS = definitely silence, skip VAD
DEFAULT_CHUNK_SILENCE = 1.0
MAX_CHUNK_SEC = 20
OVERLAP_SEC = 2.0  # seconds of audio overlap between chunks


@dataclass(frozen=True)
class AudioChunk:
    """A chunk of audio with its position in the recording timeline."""

    audio: np.ndarray
    start_time: float  # seconds from recording start
    duration: float  # seconds
    source: str = "mic"  # "mic" or "browser"


class MeetingRecorder:
    """Records audio continuously and splits into chunks at natural pauses.

    Chunks are emitted to a queue for parallel transcription while
    recording continues. Adjacent chunks overlap by OVERLAP_SEC to
    preserve context at boundaries.
    """

    def __init__(
        self,
        chunk_queue: queue.Queue[AudioChunk | None],
        device: str | int | None = None,
        chunk_silence: float = DEFAULT_CHUNK_SILENCE,
        source_label: str = "mic",
    ) -> None:
        self._chunk_queue = chunk_queue
        self._device = device
        self._chunk_silence = chunk_silence
        self._source_label = source_label

        self._vad = SileroVAD()
        self._channels = self._detect_channels()

        self._stream: sd.InputStream | None = None
        self._frames: list[np.ndarray] = []
        self._recording = False

        self._recording_start: float = 0.0
        self._chunk_start: float = 0.0
        self._voice_detected_in_chunk = False
        self._silence_start: float | None = None
        self._vad_buffer = np.array([], dtype=np.float32)

    def _reset_state(self) -> None:
        """Reset all recording state for a new session."""
        self._frames = []
        self._recording = True
        self._recording_start = time.monotonic()
        self._chunk_start = 0.0
        self._voice_detected_in_chunk = False
        self._silence_start = None
        self._vad_buffer = np.array([], dtype=np.float32)
        self._vad.reset_states()

    def start(self) -> None:
        """Start continuous recording from an audio device."""
        self._reset_state()

        self._stream = sd.InputStream(
            device=self._device,
            samplerate=SAMPLE_RATE,
            channels=self._channels,
            dtype=DTYPE,
            blocksize=BLOCK_SIZE,
            callback=self._callback,
        )
        self._stream.start()

        device_name = self._device or "system default"
        logger.info(
            "Meeting recording started (device: %s, channels: %d)",
            device_name, self._channels,
        )

    def start_without_device(self) -> None:
        """Start recording state without an audio device.

        Use with feed_audio() for external audio sources (e.g. WebSocket).
        """
        self._reset_state()
        logger.info("Meeting recording started (external audio source)")

    def stop(self) -> None:
        """Stop recording from audio device and emit remaining audio."""
        self._recording = False

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        self._finalize()

    def stop_without_device(self) -> None:
        """Stop recording state for external audio sources."""
        self._recording = False
        self._finalize()

    def _finalize(self) -> None:
        """Emit remaining audio, send sentinel, reset VAD."""
        self._emit_chunk(is_final=True)
        self._chunk_queue.put(None)
        self._vad.reset_states()

        elapsed = time.monotonic() - self._recording_start
        logger.info("Meeting recording stopped — total %.1fs", elapsed)

    def feed_audio(self, samples: np.ndarray) -> None:
        """Feed externally-sourced audio into the recording pipeline.

        Accepts a 1D float32 mono array of arbitrary length. Buffers
        internally and processes in BLOCK_SIZE (512-sample) chunks
        through VAD and chunk-splitting logic.
        """
        if not self._recording:
            return

        self._vad_buffer = np.concatenate([self._vad_buffer, samples])

        while len(self._vad_buffer) >= BLOCK_SIZE:
            block = self._vad_buffer[:BLOCK_SIZE]
            self._vad_buffer = self._vad_buffer[BLOCK_SIZE:]
            self._process_block(block)

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

        if indata.shape[1] > 1:
            mono = indata.mean(axis=1, keepdims=True)
        else:
            mono = indata

        self._process_block(mono[:, 0].copy())

    def _process_block(self, samples: np.ndarray) -> None:
        """Process a single block of mono audio through VAD and chunk splitting."""
        self._frames.append(samples.reshape(-1, 1))

        chunk_elapsed = self._current_chunk_duration()
        if chunk_elapsed >= MAX_CHUNK_SEC:
            logger.info("Chunk max duration reached (%.0fs), splitting", MAX_CHUNK_SEC)
            self._emit_chunk()
            return

        rms = np.sqrt(np.mean(samples ** 2))
        if rms < RMS_SILENCE_FLOOR:
            speech_prob = 0.0
        else:
            speech_prob = self._vad(samples.copy())
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

    def _emit_chunk(self, is_final: bool = False) -> None:
        """Concatenate accumulated frames and put chunk on queue."""
        if not self._frames:
            return

        audio = np.concatenate(self._frames, axis=0).squeeze()
        duration = len(audio) / SAMPLE_RATE

        if duration < 0.3 and not is_final:
            self._frames = []
            return

        # Skip silent chunks — no voice detected means only background noise.
        # Whisper hallucinates on silence (e.g. "Thank you.", random words).
        if not self._voice_detected_in_chunk and not is_final:
            logger.debug("Skipping silent chunk (%.1fs, no voice detected)", duration)
            self._frames = []
            self._chunk_start += duration
            self._silence_start = None
            self._vad.reset_states()
            return

        chunk = AudioChunk(
            audio=audio,
            start_time=self._chunk_start,
            duration=duration,
            source=self._source_label,
        )
        self._chunk_queue.put(chunk)
        logger.info(
            "Chunk emitted: start=%.1fs, duration=%.1fs",
            chunk.start_time, chunk.duration,
        )

        # Keep overlap samples for next chunk (preserves context at boundary)
        overlap_samples = int(OVERLAP_SEC * SAMPLE_RATE)
        if not is_final and len(audio) > overlap_samples:
            overlap_audio = audio[-overlap_samples:]
            self._frames = [overlap_audio.reshape(-1, 1)]
            self._chunk_start += duration - OVERLAP_SEC
        else:
            self._frames = []
            self._chunk_start += duration

        self._voice_detected_in_chunk = False
        self._silence_start = None
        self._vad.reset_states()

    def _detect_channels(self) -> int:
        """Detect the number of input channels for the selected device."""
        if self._device is None:
            return CHANNELS
        try:
            info = sd.query_devices(self._device)
            ch = info["max_input_channels"]
            if ch > 1:
                logger.info("Device '%s' has %d input channels (will mix to mono)", self._device, ch)
            return max(ch, 1)
        except Exception:
            return CHANNELS

    def _current_chunk_duration(self) -> float:
        total_samples = sum(f.shape[0] for f in self._frames)
        return total_samples / SAMPLE_RATE
