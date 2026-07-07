"""Audio recording with Silero VAD for automatic silence detection."""

import logging
import queue
import time

import numpy as np
import sounddevice as sd

from whisper_daemon import audio_system, telemetry
from whisper_daemon.events import Event, EventType
from whisper_daemon.vad import SileroVAD

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
DTYPE = "float32"
BLOCK_SIZE = 512  # Matches Silero VAD expected chunk size (32ms at 16kHz)
MAX_RECORDING_SEC = 120
NO_VOICE_TIMEOUT_SEC = 8.0  # Bail out if VAD never detects voice — likely dead/muted input
VAD_THRESHOLD = 0.5
RMS_SILENCE_FLOOR = 0.005  # Below this RMS = definitely silence, skip VAD
DEFAULT_SILENCE_SEC = 0.7
PREVIEW_START_SEC = 2.0  # Start previews after 2s of voice
PREVIEW_INTERVAL_SEC = 3.0  # Send preview every 3s
# A live stream delivering exact-zero samples is a dead device (stale
# PortAudio binding after a Bluetooth topology change, revoked mic
# permission, OS-level mute). A real quiet room always has ~1e-6 noise.
# 63 blocks × 512 samples ÷ 16 kHz ≈ 2 s.
DEAD_STREAM_BLOCKS = 63


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
        self._zero_block_count = 0
        self._device_dead_reported = False

    def start_recording(self) -> None:
        """Start capturing audio. Falls back to system default if preferred device fails."""
        self._chunks = []
        self._voice_detected = False
        self._silence_start = None
        self._recording_start = time.monotonic()
        self._recording = True
        self._last_preview_time = 0.0
        self._last_preview_samples = 0
        self._zero_block_count = 0
        self._device_dead_reported = False
        self._vad.reset_states()

        with audio_system.stream_open():
            audio_system.ensure_fresh()
            device, channels = self._open_stream()
            self._stream.start()
            audio_system.register_stream(self._stream)
        logger.info("Recording started (device: %s, channels: %d)", device or "system default", channels)

    def reopen_stream(self) -> bool:
        """Replace a dead stream mid-recording, keeping buffered audio.

        Called by the daemon when the callback reports a dead device
        (all-zero samples). Closes the old stream, refreshes the PortAudio
        device list, and opens a new stream so the recording continues
        transparently. Returns True on success.
        """
        self._close_stream()
        audio_system.mark_dirty("dead stream during dictation")
        with audio_system.stream_open():
            audio_system.ensure_fresh()
            try:
                device, channels = self._open_stream()
                self._stream.start()
            except Exception:
                logger.exception("Stream reopen failed")
                # Don't leave a half-open stream behind — close it so it
                # can't linger unregistered and keep the mic hot.
                self._close_stream()
                return False
            audio_system.register_stream(self._stream)
        self._zero_block_count = 0
        self._device_dead_reported = False
        # Extend the no-voice deadline — the user gets a fresh window on
        # the recovered device instead of an instant abort.
        self._recording_start = time.monotonic()
        logger.info(
            "Recording stream reopened (device: %s, channels: %d)",
            device or "system default", channels,
        )
        return True

    def _close_stream(self) -> None:
        if self._stream is None:
            return
        audio_system.unregister_stream(self._stream)
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            logger.debug("Error closing stream", exc_info=True)
        self._stream = None

    def _open_stream(self) -> tuple[str | int | None, int]:
        """Open the input stream, preferring the configured device.

        Any open failure means the PortAudio device snapshot is suspect:
        mark it dirty and refresh (refused automatically while another
        stream — e.g. a meeting recording — is live), retry the preferred
        device once on a fresh list, then fall back to the system default.
        """
        if self._device is not None:
            try:
                return self._try_open(self._device)
            except (sd.PortAudioError, ValueError):
                audio_system.mark_dirty(f"device '{self._device}' failed to open")
                if audio_system.ensure_fresh():
                    try:
                        return self._try_open(self._device)
                    except (sd.PortAudioError, ValueError):
                        logger.warning(
                            "Device '%s' still unavailable after refresh",
                            self._device,
                        )
                logger.warning(
                    "Device '%s' unavailable, falling back to system default",
                    self._device,
                )

        try:
            return self._try_open(None)
        except sd.PortAudioError:
            audio_system.mark_dirty("system default failed to open")
            logger.warning(
                "System default also failed — refreshing PortAudio device list",
            )
            if not audio_system.ensure_fresh():
                raise
        return self._try_open(None)

    def _try_open(self, device: str | int | None) -> tuple[str | int | None, int]:
        self._channels = _detect_channels(device) if device is not None else 1
        self._stream = sd.InputStream(
            device=device,
            samplerate=SAMPLE_RATE,
            channels=self._channels,
            dtype=DTYPE,
            blocksize=BLOCK_SIZE,
            callback=self._audio_callback,
        )
        return device, self._channels

    @property
    def voice_detected(self) -> bool:
        """Whether VAD detected voice at any point during the last recording."""
        return self._voice_detected

    def stop_recording(self) -> np.ndarray:
        """Stop recording and return the captured audio as a 1D numpy array."""
        self._recording = False

        self._close_stream()

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

        # Dead-stream detection: exact-zero samples mean the device behind
        # this stream is gone (stale binding after a Bluetooth reconnect,
        # revoked permission). Report once; the daemon reopens the stream
        # on a refreshed device list and the recording continues.
        if np.all(mono[:, 0] == 0.0):
            self._zero_block_count += 1
            if (
                self._zero_block_count >= DEAD_STREAM_BLOCKS
                and not self._device_dead_reported
            ):
                self._device_dead_reported = True
                logger.warning(
                    "Stream delivering only zeros for ~%.1fs — device dead, requesting reopen",
                    self._zero_block_count * BLOCK_SIZE / SAMPLE_RATE,
                )
                self._queue.put(Event(EventType.RECORD_DEVICE_DEAD))
        else:
            self._zero_block_count = 0

        elapsed = time.monotonic() - self._recording_start
        if elapsed > MAX_RECORDING_SEC:
            logger.warning("Max recording time reached (%ds)", MAX_RECORDING_SEC)
            self._queue.put(Event(EventType.RECORD_STOP))
            return

        # Dead-mic guard: if VAD never saw voice after a reasonable window,
        # the input device is probably silent (unplugged BT headset, muted
        # default input, etc). Stopping here avoids recording indefinitely
        # and prevents Whisper from hallucinating English phrases on silence.
        if not self._voice_detected and elapsed > NO_VOICE_TIMEOUT_SEC:
            logger.warning(
                "No voice detected after %.1fs — aborting recording (check microphone)",
                NO_VOICE_TIMEOUT_SEC,
            )
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
