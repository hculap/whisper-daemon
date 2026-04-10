"""Audio recording with Silero VAD for automatic silence detection."""

import logging
import queue
import time

import numpy as np
import sounddevice as sd

from whisper_daemon import telemetry
from whisper_daemon.events import Event, EventType
from whisper_daemon.vad import SileroVAD

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
DTYPE = "float32"
BLOCK_SIZE = 512  # Matches Silero VAD expected chunk size (32ms at 16kHz)
MAX_RECORDING_SEC = 120
VAD_THRESHOLD = 0.5
RMS_SILENCE_FLOOR = 0.005  # Below this RMS = definitely silence, skip VAD
DEFAULT_SILENCE_SEC = 0.7
PREVIEW_START_SEC = 2.0  # Start previews after 2s of voice
PREVIEW_INTERVAL_SEC = 3.0  # Send preview every 3s


class AudioRecorder:
    """Records audio with Silero VAD auto-stop. Supports multi-channel devices."""

    def __init__(
        self,
        event_queue: queue.Queue[Event],
        silence_timeout: float = DEFAULT_SILENCE_SEC,
        device: str | int | None = None,
    ) -> None:
        self._queue = event_queue
        self._silence_timeout = silence_timeout
        self._device = device
        self._channels = _detect_channels(device)

        self._vad = SileroVAD()

        self._stream: sd.InputStream | None = None
        self._chunks: list[np.ndarray] = []
        self._voice_detected = False
        self._silence_start: float | None = None
        self._recording_start: float = 0.0
        self._recording = False
        self._last_preview_time: float = 0.0
        self._last_preview_samples: int = 0

    def start_recording(self) -> None:
        """Start capturing audio. Falls back to system default if preferred device fails."""
        self._chunks = []
        self._voice_detected = False
        self._silence_start = None
        self._recording_start = time.monotonic()
        self._recording = True
        self._last_preview_time = 0.0
        self._last_preview_samples = 0
        self._vad.reset_states()

        device, channels = self._open_stream()
        self._stream.start()
        logger.info("Recording started (device: %s, channels: %d)", device or "system default", channels)

    def _open_stream(self) -> tuple[str | int | None, int]:
        """Try preferred device, fall back to system default on failure.

        Note: previously this called ``sd._terminate()`` / ``sd._initialize()``
        to refresh the PortAudio device list for hot-plugged Bluetooth/USB
        mics. That turns out to be global: it tears down the whole host
        API, destroying any other live ``InputStream`` (e.g. a running
        meeting recording). The ``PortAudioError`` fallback to
        ``device=None`` below already handles "preferred device
        disappeared" — which is the only real-world case — so we skip the
        refresh and keep the other recorders alive.
        """
        if self._device is not None:
            try:
                self._channels = _detect_channels(self._device)
                self._stream = sd.InputStream(
                    device=self._device,
                    samplerate=SAMPLE_RATE,
                    channels=self._channels,
                    dtype=DTYPE,
                    blocksize=BLOCK_SIZE,
                    callback=self._audio_callback,
                )
                return self._device, self._channels
            except sd.PortAudioError:
                logger.warning(
                    "Device '%s' unavailable, falling back to system default",
                    self._device,
                )

        self._channels = 1
        self._stream = sd.InputStream(
            device=None,
            samplerate=SAMPLE_RATE,
            channels=self._channels,
            dtype=DTYPE,
            blocksize=BLOCK_SIZE,
            callback=self._audio_callback,
        )
        return None, self._channels

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

    def get_audio_snapshot(self) -> tuple[np.ndarray, int]:
        """Get a copy of the current audio buffer without stopping recording.

        Returns (audio_array, total_samples) for progressive transcription.
        """
        if not self._chunks:
            return np.array([], dtype=np.float32), 0
        audio = np.concatenate(self._chunks, axis=0).squeeze()
        return audio, len(audio)

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

        # Mix to mono if multi-channel
        if indata.shape[1] > 1:
            mono = indata.mean(axis=1, keepdims=True)
        else:
            mono = indata
        self._chunks.append(mono.copy())

        elapsed = time.monotonic() - self._recording_start
        if elapsed > MAX_RECORDING_SEC:
            logger.warning("Max recording time reached (%ds)", MAX_RECORDING_SEC)
            self._queue.put(Event(EventType.RECORD_STOP))
            return

        # RMS energy pre-filter: skip VAD for obviously silent frames
        samples = mono[:, 0]
        rms = np.sqrt(np.mean(samples ** 2))
        if rms < RMS_SILENCE_FLOOR:
            speech_prob = 0.0
        else:
            speech_prob = self._vad(samples.copy())

        now = time.monotonic()

        if speech_prob > VAD_THRESHOLD:
            self._voice_detected = True
            self._silence_start = None
        elif self._voice_detected:
            if self._silence_start is None:
                self._silence_start = now
            elif now - self._silence_start >= self._silence_timeout:
                telemetry.mark("vad_silence")
                logger.info("VAD: silence detected (%.1fs)", self._silence_timeout)
                self._queue.put(Event(EventType.RECORD_STOP))
                return

        # Progressive transcription: emit preview snapshots while recording
        if self._voice_detected and elapsed > PREVIEW_START_SEC:
            time_since_preview = now - self._last_preview_time
            if self._last_preview_time == 0.0 or time_since_preview >= PREVIEW_INTERVAL_SEC:
                total_samples = sum(c.shape[0] for c in self._chunks)
                if total_samples > self._last_preview_samples:
                    self._last_preview_time = now
                    self._last_preview_samples = total_samples
                    self._queue.put(Event(EventType.TRANSCRIBE_PREVIEW))


def _detect_channels(device: str | int | None) -> int:
    """Detect the number of input channels for the selected device."""
    if device is None:
        return 1
    try:
        info = sd.query_devices(device)
        ch = info["max_input_channels"]
        if ch > 1:
            logger.info("Device '%s' has %d input channels (will mix to mono)", device, ch)
        return max(ch, 1)
    except Exception:
        return 1
