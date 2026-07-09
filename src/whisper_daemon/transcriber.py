"""Whisper transcription using mlx-whisper."""

import logging
import os
import threading

import mlx_whisper
import numpy as np

from whisper_daemon.ffmpeg_path import ensure_ffmpeg_on_path

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo-q4"

# MLX Metal calls are not concurrency-safe: two inferences at once
# (dictation vs meeting chunk vs file transcription) can SIGABRT or hang
# the GPU queue. Every mlx_whisper call in the process goes through this
# lock. RLock so a caller that already holds it (the dictation daemon's
# timeout-guarded acquire) can call transcribe() without deadlocking.
MLX_LOCK = threading.RLock()

_model_preloaded = False


def _hang_watchdog(audio_sec: float, timeout_sec: float | None = None) -> threading.Timer:
    """Arm a timer that force-restarts the daemon if inference wedges.

    A hung MLX call is unrecoverable in-process: the GPU queue is stuck
    and MLX_LOCK is held forever, so every later transcription would
    fail too. MLX runs ~1-2x realtime on Apple Silicon; 5x + 60s covers
    GPU pressure comfortably. Pass ``timeout_sec`` to override (e.g. a
    fixed budget for warmup where there is no audio length). The caller
    must cancel the timer when the inference returns.
    """
    from whisper_daemon import supervisor

    if timeout_sec is None:
        timeout_sec = max(60.0, audio_sec * 5.0 + 60.0)

    def _fire() -> None:
        logger.critical(
            "MLX inference hung >%.0fs (audio=%.1fs) — restarting daemon",
            timeout_sec, audio_sec,
        )
        supervisor.request_restart("MLX inference hung")

    timer = threading.Timer(timeout_sec, _fire)
    timer.daemon = True
    timer.start()
    return timer


def preload_model(model: str = DEFAULT_MODEL) -> None:
    """Download and load the model into GPU memory at startup."""
    global _model_preloaded
    logger.info("Preloading model %s...", model)
    old_hf_verbosity = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    try:
        warmup_audio = np.zeros(16000, dtype=np.float32)
        with MLX_LOCK:
            # Includes a possible model download on first run — give it a
            # generous fixed budget rather than the audio-length formula.
            watchdog = _hang_watchdog(1.0, timeout_sec=600.0)
            try:
                mlx_whisper.transcribe(warmup_audio, path_or_hf_repo=model)
            finally:
                watchdog.cancel()
    finally:
        if old_hf_verbosity is None:
            os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
        else:
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = old_hf_verbosity
    _model_preloaded = True
    logger.info("Model preloaded and ready")


def transcribe(audio: np.ndarray, model: str = DEFAULT_MODEL) -> str:
    """Transcribe audio array to text using mlx-whisper.

    Args:
        audio: 1D float32 numpy array, 16kHz mono.
        model: HuggingFace model repo ID.

    Returns:
        Transcribed text (stripped), or empty string on failure.
    """
    if audio.size == 0:
        logger.warning("Empty audio buffer, skipping transcription")
        return ""

    # Pad very short audio to 1s — Whisper can hang in decode loops on tiny clips
    min_samples = 16000
    if len(audio) < min_samples:
        audio = np.pad(audio, (0, min_samples - len(audio)))

    try:
        logger.info("Transcribing %.1fs of audio...", len(audio) / 16000)
        with MLX_LOCK:
            watchdog = _hang_watchdog(len(audio) / 16000)
            try:
                result = mlx_whisper.transcribe(
                    audio,
                    path_or_hf_repo=model,
                    temperature=0,
                    condition_on_previous_text=False,
                    word_timestamps=False,
                )
            finally:
                watchdog.cancel()
        text = result.get("text", "").strip()
        language = result.get("language", "unknown")
        logger.info("Transcription done — lang=%s, len=%d chars", language, len(text))
        return text
    except Exception:
        logger.exception("Transcription failed")
        return ""


def transcribe_full(
    audio: np.ndarray,
    model: str = DEFAULT_MODEL,
    language: str | None = None,
) -> dict:
    """Transcribe audio array and return the full result dict with segments.

    Args:
        audio: 1D float32 numpy array, 16kHz mono.
        model: HuggingFace model repo ID.
        language: Force a language code, or None to auto-detect. Forcing avoids
            per-chunk language misdetection (short/quiet meeting chunks otherwise
            come back as Icelandic/Russian/etc.).

    Returns:
        Dict with keys: text, segments, language.
    """
    if audio.size == 0:
        return {"text": "", "segments": [], "language": ""}

    # Pad very short audio to 1s — Whisper can hang in decode loops on tiny clips
    min_samples = 16000
    if len(audio) < min_samples:
        audio = np.pad(audio, (0, min_samples - len(audio)))

    kwargs: dict = {
        "path_or_hf_repo": model,
        "temperature": 0,
        "condition_on_previous_text": False,
        "word_timestamps": False,
    }
    if language:
        kwargs["language"] = language

    try:
        logger.info("Transcribing %.1fs of audio...", len(audio) / 16000)
        with MLX_LOCK:
            watchdog = _hang_watchdog(len(audio) / 16000)
            try:
                result = mlx_whisper.transcribe(audio, **kwargs)
            finally:
                watchdog.cancel()
        # Strip leading whitespace from segment texts
        for seg in result.get("segments", []):
            seg["text"] = seg["text"].strip()
        result["text"] = result.get("text", "").strip()

        logger.info(
            "Transcription done — lang=%s, %d chars, %d segments",
            result.get("language", "unknown"),
            len(result.get("text", "")),
            len(result.get("segments", [])),
        )
        return result
    except Exception:
        logger.exception("Transcription failed")
        return {"text": "", "segments": [], "language": ""}


def transcribe_file(
    path: str,
    model: str = DEFAULT_MODEL,
    language: str | None = None,
) -> dict:
    """Transcribe an audio/video file and return the full result dict.

    Args:
        path: Path to audio/video file (ffmpeg handles decoding).
        model: HuggingFace model repo ID.
        language: Force language code, or None for auto-detect.

    Returns:
        Dict with keys: text, segments, language.
    """
    logger.info("Transcribing file: %s", path)
    kwargs: dict = {
        "path_or_hf_repo": model,
        "temperature": 0,
        "condition_on_previous_text": False,
        "word_timestamps": False,
    }
    if language is not None:
        kwargs["language"] = language

    # ffmpeg decodes the file; under launchd the daemon's PATH excludes
    # Homebrew/MacPorts, so make sure the binary is discoverable first and
    # fail with an actionable message instead of a bare FileNotFoundError.
    if ensure_ffmpeg_on_path() is None:
        raise RuntimeError(
            "ffmpeg not found — it is required to decode audio/video files "
            "(m4a, mp4, mp3, …). Install it with: brew install ffmpeg"
        )

    # Decode to an array BEFORE taking the GPU lock: a stalled ffmpeg on a
    # corrupt file then can't block dictation/meeting inference, and the
    # decoded length gives the hang watchdog an exact duration to budget.
    audio = mlx_whisper.audio.load_audio(path)
    audio_sec = len(audio) / 16000

    with MLX_LOCK:
        watchdog = _hang_watchdog(audio_sec)
        try:
            result = mlx_whisper.transcribe(audio, **kwargs)
        finally:
            watchdog.cancel()
    language_detected = result.get("language", "unknown")
    text_len = len(result.get("text", ""))
    segments_count = len(result.get("segments", []))
    logger.info(
        "Done — lang=%s, %d chars, %d segments",
        language_detected, text_len, segments_count,
    )
    return result
