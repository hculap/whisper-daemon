"""Text-to-speech via Piper TTS with automatic language detection."""

import logging
import subprocess
import tempfile
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

PIPER_BIN = Path.home() / "Projects" / "Utils" / "whisper-daemon" / ".venv" / "bin" / "piper"

VOICES = {
    "pl": Path.home() / ".local" / "share" / "piper" / "pl_PL-gosia-medium.onnx",
    "en": Path.home() / ".local" / "share" / "piper" / "en_US-amy-medium.onnx",
}

POLISH_CHARS = set("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ")

_speaking_lock = threading.Lock()
_stop_event = threading.Event()


def _detect_language(text: str) -> str:
    polish_count = sum(1 for ch in text if ch in POLISH_CHARS)
    return "pl" if polish_count >= 2 else "en"


def speak_text(text: str) -> None:
    """Speak text using Piper TTS in a background thread."""
    if not text.strip():
        return

    thread = threading.Thread(target=_speak_worker, args=(text,), daemon=True)
    thread.start()


def stop_speaking() -> None:
    """Signal any in-progress TTS to stop."""
    _stop_event.set()


def _speak_worker(text: str) -> None:
    if not _speaking_lock.acquire(blocking=False):
        logger.info("TTS already speaking, skipping")
        return

    _stop_event.clear()
    try:
        lang = _detect_language(text)
        voice = VOICES.get(lang, VOICES["en"])

        if not voice.exists():
            logger.error("Voice model not found: %s", voice)
            return

        if not PIPER_BIN.exists():
            logger.error("Piper binary not found: %s", PIPER_BIN)
            return

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name

        logger.info("TTS: lang=%s, voice=%s, %d chars", lang, voice.name, len(text))

        piper_proc = subprocess.run(
            [str(PIPER_BIN), "--model", str(voice), "--output_file", wav_path],
            input=text,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if piper_proc.returncode != 0:
            logger.error("Piper failed: %s", piper_proc.stderr)
            return

        if _stop_event.is_set():
            return

        subprocess.run(["afplay", wav_path], timeout=60)

        Path(wav_path).unlink(missing_ok=True)
        logger.info("TTS playback complete")

    except subprocess.TimeoutExpired:
        logger.error("TTS timed out")
    except Exception:
        logger.exception("TTS failed")
    finally:
        _speaking_lock.release()
