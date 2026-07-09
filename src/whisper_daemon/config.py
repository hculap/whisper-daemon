"""Persistent settings stored in ~/.config/whisper-daemon/config.toml."""

import logging
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

CONFIG_DIR = Path.home() / ".config" / "whisper-daemon"
CONFIG_FILE = CONFIG_DIR / "config.toml"

VALID_FORMATS = {"txt", "srt", "vtt", "json"}
VALID_SCREENSHOT_DISPLAYS = {"all", "primary"}
# "auto" = let Whisper detect per chunk (unreliable on short/quiet audio, which
# is why forcing a language exists). The codes are a curated subset of Whisper's
# supported languages, enough for the common meeting cases.
VALID_LANGUAGES = {
    "auto", "en", "pl", "de", "es", "fr", "it", "pt", "nl",
    "uk", "ru", "cs", "sk", "sv", "no", "da", "fi",
}


def _validate_screenshot_displays(value: str) -> str:
    """Coerce screenshot_displays to a valid value, defaulting to 'all'."""
    return value if value in VALID_SCREENSHOT_DISPLAYS else "all"


def _validate_language(value: str) -> str:
    """Coerce a transcription language to a valid code, defaulting to 'auto'."""
    return value if value in VALID_LANGUAGES else "auto"


_TOML_SIMPLE_ESCAPES = {
    "\\": "\\\\",
    '"': '\\"',
    "\b": "\\b",
    "\t": "\\t",
    "\n": "\\n",
    "\f": "\\f",
    "\r": "\\r",
}


def _toml_str(value: str) -> str:
    """Serialize a Python string as a safe TOML basic string.

    Escapes per the TOML spec — backslash, double-quote, the named control
    escapes (\\b \\t \\n \\f \\r) and any remaining control character as
    ``\\uXXXX`` — so free-form values (e.g. recording_dir/device coming from
    the extension) cannot inject extra keys or corrupt config.toml even if a
    value slips past upstream validation. Lone UTF-16 surrogate codepoints
    (``U+D800``–``U+DFFF``) are dropped: they are not valid Unicode scalars,
    are not utf-8 encodable, and a bare ``\\uD800`` TOML escape is itself
    invalid — so emitting them would still corrupt the file on write.
    """
    out: list[str] = []
    for ch in str(value):
        escape = _TOML_SIMPLE_ESCAPES.get(ch)
        if escape is not None:
            out.append(escape)
        elif 0xD800 <= ord(ch) <= 0xDFFF:
            continue  # drop lone surrogate — not utf-8 encodable
        elif ord(ch) < 0x20 or ord(ch) == 0x7F:
            out.append(f"\\u{ord(ch):04X}")
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


@dataclass
class Settings:
    recording_dir: str = "~/Desktop"
    recording_formats: list[str] = field(default_factory=lambda: ["txt"])
    recording_device: str = ""  # empty = system default
    capture_mic: bool = True  # capture local microphone during meetings
    capture_tab: bool = True  # capture browser tab audio during meetings
    live_captions: bool = False  # show live captions in the extension UI
    recording_language: str = "auto"  # forced transcription language; "auto" = detect
    screenshot_displays: str = "all"  # "all" or "primary"
    save_audio: bool = False  # save raw audio alongside transcripts
    capture_screenshots: bool = False  # capture screenshots during recording
    screenshot_interval: float = 30.0  # seconds between fallback captures
    screenshot_event_triggers: bool = True  # capture on keyboard/mouse/scroll
    screenshot_debounce: float = 2.0  # seconds after last event before capture
    screenshot_cooldown: float = 5.0  # minimum seconds between captures
    diarize: bool = False  # enable speaker diarization for meeting recordings
    diarize_mode: str = "hybrid"  # batch, realtime, or hybrid
    auto_record_meetings: bool = True  # auto-start recording on browser meeting detect
    tts_language: str = "auto"  # auto, pl, or en
    transcription_formats: list[str] = field(default_factory=lambda: ["txt"])
    transcription_output_dir: str = ""  # empty = same as input file
    server_host: str = "127.0.0.1"
    server_port: int = 9876

    @property
    def recording_dir_path(self) -> Path:
        return Path(self.recording_dir).expanduser()

    @property
    def transcription_output_dir_path(self) -> Path | None:
        if not self.transcription_output_dir:
            return None
        return Path(self.transcription_output_dir).expanduser()


def load_settings() -> Settings:
    """Load settings from config.toml, or return defaults."""
    if not CONFIG_FILE.exists():
        return Settings()

    try:
        with open(CONFIG_FILE, "rb") as f:
            data = tomllib.load(f)

        rec = data.get("recording", {})
        trans = data.get("transcription", {})
        srv = data.get("server", {})

        return Settings(
            recording_dir=rec.get("dir", "~/Desktop"),
            recording_formats=_validate_formats(rec.get("formats", ["txt"])),
            recording_device=rec.get("device", ""),
            capture_mic=rec.get("capture_mic", True),
            capture_tab=rec.get("capture_tab", True),
            live_captions=rec.get("live_captions", False),
            recording_language=_validate_language(rec.get("language", "auto")),
            screenshot_displays=_validate_screenshot_displays(
                rec.get("screenshot_displays", "all")
            ),
            save_audio=rec.get("save_audio", False),
            capture_screenshots=rec.get("capture_screenshots", False),
            screenshot_interval=rec.get("screenshot_interval", 30.0),
            screenshot_event_triggers=rec.get("screenshot_event_triggers", True),
            screenshot_debounce=rec.get("screenshot_debounce", 2.0),
            screenshot_cooldown=rec.get("screenshot_cooldown", 5.0),
            diarize=rec.get("diarize", False),
            diarize_mode=rec.get("diarize_mode", "hybrid"),
            auto_record_meetings=rec.get("auto_record_meetings", True),
            tts_language=data.get("tts", {}).get("language", "auto"),
            transcription_formats=_validate_formats(trans.get("formats", ["txt"])),
            transcription_output_dir=trans.get("output_dir", ""),
            server_host=srv.get("host", "127.0.0.1"),
            server_port=srv.get("port", 9876),
        )
    except Exception:
        logger.exception("Failed to load config, using defaults")
        return Settings()


def save_settings(settings: Settings) -> None:
    """Save settings to config.toml."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    rec_formats = ", ".join(_toml_str(f) for f in settings.recording_formats)
    trans_formats = ", ".join(_toml_str(f) for f in settings.transcription_formats)

    lines = [
        "[recording]",
        f"dir = {_toml_str(settings.recording_dir)}",
        f"formats = [{rec_formats}]",
        f"device = {_toml_str(settings.recording_device)}",
        f'capture_mic = {"true" if settings.capture_mic else "false"}',
        f'capture_tab = {"true" if settings.capture_tab else "false"}',
        f'live_captions = {"true" if settings.live_captions else "false"}',
        f"language = {_toml_str(settings.recording_language)}",
        f"screenshot_displays = {_toml_str(settings.screenshot_displays)}",
        f'save_audio = {"true" if settings.save_audio else "false"}',
        f'capture_screenshots = {"true" if settings.capture_screenshots else "false"}',
        f"screenshot_interval = {settings.screenshot_interval}",
        f'screenshot_event_triggers = {"true" if settings.screenshot_event_triggers else "false"}',
        f"screenshot_debounce = {settings.screenshot_debounce}",
        f"screenshot_cooldown = {settings.screenshot_cooldown}",
        f'diarize = {"true" if settings.diarize else "false"}',
        f"diarize_mode = {_toml_str(settings.diarize_mode)}",
        f'auto_record_meetings = {"true" if settings.auto_record_meetings else "false"}',
        "",
        "[transcription]",
        f"formats = [{trans_formats}]",
        f"output_dir = {_toml_str(settings.transcription_output_dir)}",
        "",
        "[tts]",
        f"language = {_toml_str(settings.tts_language)}",
        "",
        "[server]",
        f"host = {_toml_str(settings.server_host)}",
        f"port = {settings.server_port}",
        "",
    ]

    CONFIG_FILE.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Settings saved to %s", CONFIG_FILE)


def _validate_formats(formats: list[str]) -> list[str]:
    valid = [f for f in formats if f in VALID_FORMATS]
    return valid if valid else ["txt"]
