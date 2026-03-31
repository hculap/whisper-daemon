"""Whisper transcription using mlx-whisper."""

import logging
import os

import mlx_whisper
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo-q4"

_model_preloaded = False


def preload_model(model: str = DEFAULT_MODEL) -> None:
    """Download and load the model into GPU memory at startup."""
    global _model_preloaded
    logger.info("Preloading model %s...", model)
    old_hf_verbosity = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    try:
        warmup_audio = np.zeros(16000, dtype=np.float32)
        mlx_whisper.transcribe(warmup_audio, path_or_hf_repo=model)
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

    try:
        logger.info("Transcribing %.1fs of audio...", len(audio) / 16000)
        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=model,
            temperature=0,
            condition_on_previous_text=False,
            word_timestamps=False,
        )
        text = result.get("text", "").strip()
        language = result.get("language", "unknown")
        logger.info("Transcription done — lang=%s, len=%d chars", language, len(text))
        return text
    except Exception:
        logger.exception("Transcription failed")
        return ""


def transcribe_full(audio: np.ndarray, model: str = DEFAULT_MODEL) -> dict:
    """Transcribe audio array and return the full result dict with segments.

    Args:
        audio: 1D float32 numpy array, 16kHz mono.
        model: HuggingFace model repo ID.

    Returns:
        Dict with keys: text, segments, language.
    """
    if audio.size == 0:
        return {"text": "", "segments": [], "language": ""}

    try:
        logger.info("Transcribing %.1fs of audio...", len(audio) / 16000)
        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=model,
            temperature=0,
            condition_on_previous_text=False,
            word_timestamps=False,
        )
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

    result = mlx_whisper.transcribe(path, **kwargs)
    language_detected = result.get("language", "unknown")
    text_len = len(result.get("text", ""))
    segments_count = len(result.get("segments", []))
    logger.info(
        "Done — lang=%s, %d chars, %d segments",
        language_detected, text_len, segments_count,
    )
    return result
