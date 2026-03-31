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
