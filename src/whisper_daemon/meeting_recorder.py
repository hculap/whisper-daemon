"""Long-form meeting recorder with VAD-based chunk splitting."""

import logging
import queue
import threading
import time
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

from whisper_daemon import audio_system
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
# Dead-stream detection: an input stream that keeps delivering exact-zero
# samples for this many consecutive blocks is assumed broken.
# 900 blocks × 512 samples ÷ 16 kHz ≈ 29 s.
DEAD_STREAM_BLOCKS = 900
# Warn on the menubar every N consecutive silent chunks dropped.
SILENT_CHUNK_WARN_EVERY = 3
# A mic stream producing this many consecutive voiceless chunks (~2 min at
# 20s chunks) is treated as dead even when samples are non-zero noise —
# the zero-block detector can't see a stale device that still delivers a
# low noise floor. Triggers a stream reopen on a fresh device list.
SILENT_CHUNK_RECOVERY_STREAK = 6
# The microphone is read in PortAudio's blocking mode by a reader thread, never
# by a callback: Python on CoreAudio's real-time thread loses audio whenever it
# waits ~30 ms for the GIL or the store's lock. PortAudio's buffer (sized by
# this latency) holds the audio while the reader is busy. CoreAudio buffers
# about twice the latency (measured: 1.0 → 2.05 s, 4.0 → 8.2 s). A real meeting
# lost up to 2.9 s at 1.0, i.e. the reader stalled for ~5 s.
READ_LATENCY = 4.0
READ_POLL_SEC = 0.01
READER_JOIN_SEC = 2.0
# Falling this far behind the wall clock means device audio was lost: move the
# track forward so the loss becomes a gap instead of shifting later audio.
REANCHOR_SAMPLES = SAMPLE_RATE // 4


@dataclass(frozen=True)
class AudioChunk:
    """A chunk of audio with its position in the recording timeline."""

    audio: np.ndarray
    start_time: float  # seconds from recording start
    duration: float  # seconds
    source: str = "mic"  # "mic" or "browser"
    job_id: str = ""


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
        raw_sink=None,
        paused=None,
        sample_filter=None,
        started_at: float | None = None,
        clock=time.monotonic,
    ) -> None:
        self._chunk_queue = chunk_queue
        self._device = device
        self._chunk_silence = chunk_silence
        self._source_label = source_label
        self._raw_sink = raw_sink
        self._paused = paused or (lambda: False)
        self._sample_filter = sample_filter
        self._vad_position = None
        self.error = None
        self._capture_lock = threading.RLock()
        # Session clock origin for device positions (the store's started_at).
        self._started_at = started_at
        self._clock = clock
        self._next_position: int | None = None
        self._overflows = 0
        self._reader: threading.Thread | None = None
        self._reader_stop: threading.Event | None = None

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
        self._device_error_count: int = 0
        self._needs_recovery: bool = False
        self._recovery_attempts: int = 0
        self._fell_back_to_default: bool = False
        self._consecutive_silent_chunks: int = 0
        self._zero_block_count: int = 0
        self._silent_recovery_done: bool = False
        self._last_audio_at: float = 0.0

    def _reset_state(self) -> None:
        """Reset all recording state for a new session."""
        self._frames = []
        self._recording = True
        self._recording_start = time.monotonic()
        self._chunk_start = 0.0
        self._voice_detected_in_chunk = False
        self._silence_start = None
        self._vad_buffer = np.array([], dtype=np.float32)
        self._vad_position = None
        self._device_error_count = 0
        self._needs_recovery = False
        self._recovery_attempts = 0
        self._fell_back_to_default = False
        self._consecutive_silent_chunks = 0
        self._zero_block_count = 0
        self._silent_recovery_done = False
        self._next_position = None
        self._overflows = 0
        # Seed to "now" so the health check doesn't fire before the first
        # block has had a chance to arrive.
        self._last_audio_at = time.monotonic()
        self._vad.reset_states()

    @property
    def needs_recovery(self) -> bool:
        return self._needs_recovery

    @property
    def last_audio_at(self) -> float:
        """monotonic() of the last block that delivered any samples.

        Distinct from "last voice chunk": this advances even during
        silence, so the meeting loop can tell a genuinely dead stream (no
        blocks at all) from a live mic in a quiet room. Seeded at start.
        """
        return self._last_audio_at

    @property
    def fell_back_to_default(self) -> bool:
        return self._fell_back_to_default

    def request_recovery(self) -> None:
        """Signal from outside the reader that the stream should be reopened.

        Used by the meeting loop when it detects no chunks are arriving —
        that case never raises in read(), so the reader itself never arms
        recovery.
        """
        self._needs_recovery = True

    def check_liveness(self, now: float, timeout: float = 10) -> bool:
        """Detect a device that stopped delivering audio, even while paused or transcribing."""
        if self._source_label == 'mic' and not self._needs_recovery and now - self._last_audio_at > timeout:
            self.request_recovery()
            return True
        return False

    def recover_device(self) -> bool:
        """Attempt to reopen audio stream after device failure.

        Returns True if recovery succeeded. On failure, leaves
        ``_needs_recovery`` set so the caller can retry after a backoff
        — we never give up permanently, because devices (Bluetooth,
        USB) can reappear minutes later.
        """
        self._recovery_attempts += 1

        # Stop the old stream — set _recording=False first so no block lands
        # in _frames during _emit_chunk (closing also joins the reader).
        self._recording = False
        self._close_stream()

        # Flush any buffered audio as a final chunk so speech captured
        # just before the device died is not discarded as "silent".
        self._emit_chunk(is_final=True)
        self._device_error_count = 0
        self._zero_block_count = 0

        # The stream died behind PortAudio's back — its device snapshot is
        # stale. Refresh it (refused automatically if another stream, e.g.
        # a dictation recording, is still live) so the reopen below binds
        # real hardware instead of the same dead entry.
        audio_system.mark_dirty("meeting stream recovery")

        try:
            with audio_system.stream_open():
                audio_system.ensure_fresh()
                device, channels = self._open_stream()
                self._recording = True
                self._stream.start()
                audio_system.register_stream(self._stream)
                self._start_reader()
            self._needs_recovery = False
            self._last_audio_at = time.monotonic()
            logger.info(
                "Device recovered on attempt %d (device: %s, channels: %d)",
                self._recovery_attempts,
                device or "system default",
                channels,
            )
            self._recovery_attempts = 0
            return True
        except Exception:
            # Close any half-open stream so it can't linger unregistered.
            self._recording = False
            self._close_stream()
            logger.error(
                "Device recovery failed (attempt %d) — will retry",
                self._recovery_attempts,
                exc_info=True,
            )
            # Leave _needs_recovery=True so the meeting loop retries.
            return False

    def start(self) -> None:
        """Start continuous recording. Falls back to system default if preferred device fails."""
        self._reset_state()

        with audio_system.stream_open():
            audio_system.ensure_fresh()
            device, channels = self._open_stream()
            self._stream.start()
            audio_system.register_stream(self._stream)
            self._start_reader()
        logger.info(
            "Meeting recording started (device: %s, channels: %d)",
            device or "system default", channels,
        )

    def _close_stream(self) -> None:
        # The reader goes first: it must never touch a stream being closed.
        self._stop_reader()
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
        stream — e.g. a dictation recording — is live), retry the
        preferred device once on a fresh list, then fall back to the
        system default.
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
                self._fell_back_to_default = True

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
        self._channels = self._detect_channels() if device is not None else CHANNELS
        self._stream = sd.InputStream(
            device=device,
            samplerate=SAMPLE_RATE,
            channels=self._channels,
            dtype=DTYPE,
            blocksize=BLOCK_SIZE,
            latency=READ_LATENCY,
        )
        return device, self._channels

    def start_without_device(self) -> None:
        """Start recording state without an audio device.

        Use with feed_audio() for external audio sources (e.g. WebSocket).
        """
        self._reset_state()
        logger.info("Meeting recording started (external audio source)")

    def stop(self) -> None:
        """Stop recording from audio device and emit remaining audio."""
        self._recording = False

        self._close_stream()
        if self._overflows:
            logger.warning("Input buffer overflowed %d time(s); lost audio is marked as gaps [%s]",
                           self._overflows, self._source_label)

        self._finalize()

    def abort(self) -> None:
        """Close the device stream without finalizing (error-path cleanup).

        Idempotent: safe to call after stop() (the stream is already gone)
        so a `finally` block can guarantee the stream is never leaked
        registered — a leaked registration would disable PortAudio refresh
        for the rest of the process lifetime.
        """
        self._recording = False
        self._close_stream()

    def stop_without_device(self) -> None:
        """Stop recording state for external audio sources."""
        self._recording = False
        self._finalize()

    def _finalize(self) -> None:
        """Emit remaining audio, send sentinel, reset VAD."""
        if len(self._vad_buffer):
            self._process_block(self._vad_buffer, self._vad_position)
            self._vad_buffer = np.array([], dtype=np.float32)
        self._emit_chunk(is_final=True)
        self._chunk_queue.put(None)
        self._vad.reset_states()

        elapsed = time.monotonic() - self._recording_start
        logger.info("Meeting recording stopped — total %.1fs", elapsed)

    def feed_audio(self, samples: np.ndarray, position: int | None = None) -> None:
        """Feed externally-sourced audio into the recording pipeline.

        Accepts a 1D float32 mono array of arbitrary length. Buffers
        internally and processes in BLOCK_SIZE (512-sample) chunks
        through VAD and chunk-splitting logic.
        """
        with self._capture_lock:
            if not self._recording: return
            self._last_audio_at = time.monotonic()
            parts = (self._sample_filter(samples, position)
                     if self._sample_filter and position is not None else [(position, samples)])
            for start, part in parts:
                if start is not None and self._vad_position is not None and start != self._vad_position + len(self._vad_buffer):
                    if len(self._vad_buffer): self._process_block(self._vad_buffer, self._vad_position)
                    self._vad_buffer = np.array([], dtype=np.float32)
                    self._emit_chunk(is_final=True)
                if not len(self._vad_buffer): self._vad_position = start
                self._vad_buffer = np.concatenate([self._vad_buffer, part])
                while len(self._vad_buffer) >= BLOCK_SIZE:
                    block = self._vad_buffer[:BLOCK_SIZE]
                    self._vad_buffer = self._vad_buffer[BLOCK_SIZE:]
                    self._process_block(block, self._vad_position)
                    if self._vad_position is not None: self._vad_position += BLOCK_SIZE

    # --- Device reader (blocking mode) ---

    def _start_reader(self) -> None:
        stop = threading.Event()
        self._reader_stop = stop
        self._reader = threading.Thread(target=self._read_loop, args=(self._stream, stop),
                                        name=f"{self._source_label}-reader", daemon=True)
        self._reader.start()

    def _stop_reader(self) -> None:
        """Stop and join the reader; idempotent and bounded."""
        reader, stop = self._reader, self._reader_stop
        self._reader = self._reader_stop = None
        if stop is not None:
            stop.set()
        if reader is not None and reader is not threading.current_thread():
            reader.join(READER_JOIN_SEC)
            if reader.is_alive():
                logger.warning("Audio reader did not stop within %.0fs", READER_JOIN_SEC)

    def _read_loop(self, stream: sd.InputStream, stop: threading.Event) -> None:
        """Drain the device until stopped; a broken stream arms recovery.

        Never blocks inside PortAudio: it reads only whole blocks that are
        already buffered, so closing or recovering the stream cannot hang it.
        The meeting loop, not this thread, reopens the device.
        """
        while not stop.is_set():
            try:
                if not self.drain(stream):
                    stop.wait(READ_POLL_SEC)
            except Exception:
                if stop.is_set():
                    return
                logger.error("Audio input failed — requesting recovery", exc_info=True)
                self._needs_recovery = True
                return

    def drain(self, stream: sd.InputStream) -> int:
        """Read every whole block the device has buffered; returns the block count."""
        blocks = 0
        while stream.read_available >= BLOCK_SIZE:
            data, overflowed = stream.read(BLOCK_SIZE)
            self._handle_block(data, overflowed, stream.read_available)
            blocks += 1
        return blocks

    def _handle_block(self, indata: np.ndarray, overflowed: bool, buffered: int) -> None:
        if overflowed:
            self._overflows += 1
            logger.warning("Audio input overflowed — device audio lost [%s]", self._source_label)
        samples = (indata.mean(axis=1) if indata.shape[1] > 1 else indata[:, 0]).astype(np.float32)
        position = self._block_position(len(samples), buffered, overflowed)

        # Dead-stream detection: a live PortAudio stream that keeps
        # delivering exact-zero samples indicates the OS has muted the
        # device, revoked mic privacy, or re-routed the input behind our
        # back. None of those cases raise in read(), so recovery would never
        # trigger. A quiet room always produces ~1e-6 noise, so strict zero
        # is a reliable signal.
        if np.all(samples == 0.0):
            self._zero_block_count += 1
            if self._zero_block_count >= DEAD_STREAM_BLOCKS:
                logger.error(
                    "Audio stream delivered all-zero samples for ~%.0fs — requesting recovery",
                    self._zero_block_count * BLOCK_SIZE / SAMPLE_RATE,
                )
                self._needs_recovery = True
                self._zero_block_count = 0
        else:
            self._zero_block_count = 0

        self.feed_audio(samples, position=position)

    def _block_position(self, frames: int, buffered: int, overflowed: bool) -> int:
        """Session sample position of a block just read from the device.

        Blocks follow the device's own sample count. The wall clock (minus what
        is still buffered) only anchors the first block of a stream and moves
        the track forward when audio was lost — an overflow, or falling behind
        — so the loss becomes a recorded gap. Never moves backwards, also not
        across a reopened stream: the store rejects overlapping audio.
        """
        origin = self._started_at if self._started_at is not None else self._recording_start
        wall = max(0, round((self._clock() - origin) * SAMPLE_RATE) - buffered - frames)
        expected = self._next_position
        if expected is None or overflowed or wall > expected + REANCHOR_SAMPLES:
            position = max(wall, expected or 0)
        else:
            position = expected
        self._next_position = position + frames
        return position

    def _process_block(self, samples: np.ndarray, position=None) -> None:
        """Process a single block of mono audio through VAD and chunk splitting."""
        if position is None and self._paused():
            if self._frames: self._emit_chunk(is_final=True)
            return
        if self._raw_sink is not None:
            try:
                start = self._raw_sink(samples, position) if position is not None else self._raw_sink(samples)
                expected = self._chunk_start + self._current_chunk_duration()
                if start > expected + .01:
                    self._emit_chunk(is_final=True)
                    self._chunk_start = start
                elif not self._frames:
                    self._chunk_start = start
            except Exception as exc:
                self.error = exc
                self._recording = False
                logger.exception("Cannot persist audio; stopping this source")
                return
        self._last_audio_at = time.monotonic()
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
            speech_prob = self._vad(np.pad(samples, (0, max(0, BLOCK_SIZE - len(samples)))))
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

        audio = np.concatenate(self._frames, axis=0).reshape(-1)
        duration = len(audio) / SAMPLE_RATE

        if duration < 0.3 and not is_final:
            self._frames = []
            return

        # Skip silent chunks — no voice detected means only background noise.
        # Whisper hallucinates on silence (e.g. "Thank you.", random words).
        # Applies to final flushes too: a silent tail buffer is still silence.
        if not self._voice_detected_in_chunk:
            if not is_final:
                self._consecutive_silent_chunks += 1
                if (
                    self._consecutive_silent_chunks >= SILENT_CHUNK_WARN_EVERY
                    and self._consecutive_silent_chunks % SILENT_CHUNK_WARN_EVERY == 0
                ):
                    logger.warning(
                        "Dropped %d silent chunks in a row (~%.0fs of audio) [%s] — mic may be muted, quiet, or dead",
                        self._consecutive_silent_chunks,
                        self._consecutive_silent_chunks * duration,
                        self._source_label,
                    )
                else:
                    logger.info(
                        "Skipping silent chunk (%.1fs, no voice, streak=%d) [%s]",
                        duration,
                        self._consecutive_silent_chunks,
                        self._source_label,
                    )
                # A device-backed stream stuck on voiceless noise for this
                # long is likely a stale binding: reopen on a fresh list.
                # Once per silence episode — the user may simply not be
                # speaking, so don't churn the stream every chunk.
                # feed_audio sources (browser) have no device to recover.
                if (
                    self._stream is not None
                    and self._consecutive_silent_chunks >= SILENT_CHUNK_RECOVERY_STREAK
                    and not self._needs_recovery
                    and not self._silent_recovery_done
                ):
                    logger.warning(
                        "%d consecutive voiceless chunks — requesting stream recovery",
                        self._consecutive_silent_chunks,
                    )
                    self._silent_recovery_done = True
                    self._needs_recovery = True
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
        self._consecutive_silent_chunks = 0
        self._silent_recovery_done = False
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
