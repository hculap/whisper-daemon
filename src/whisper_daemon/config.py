"""Persistent settings stored in ~/.config/whisper-daemon/config.toml."""

import logging
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

CONFIG_DIR = Path.home() / ".config" / "whisper-daemon"
CONFIG_FILE = CONFIG_DIR / "config.toml"

VALID_FORMATS = {"txt", "srt", "vtt", "json"}


@dataclass
class Settings:
    recording_dir: str = "~/Desktop"
    recording_formats: list[str] = field(default_factory=lambda: ["txt"])
    recording_device: str = ""  # empty = system default
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

    lines = [
        "[recording]",
        f'dir = "{settings.recording_dir}"',
        f'formats = [{", ".join(f"\"{f}\"" for f in settings.recording_formats)}]',
        f'device = "{settings.recording_device}"',
        f'save_audio = {"true" if settings.save_audio else "false"}',
        f'capture_screenshots = {"true" if settings.capture_screenshots else "false"}',
        f"screenshot_interval = {settings.screenshot_interval}",
        f'screenshot_event_triggers = {"true" if settings.screenshot_event_triggers else "false"}',
        f"screenshot_debounce = {settings.screenshot_debounce}",
        f"screenshot_cooldown = {settings.screenshot_cooldown}",
        f'diarize = {"true" if settings.diarize else "false"}',
        f'diarize_mode = "{settings.diarize_mode}"',
        f'auto_record_meetings = {"true" if settings.auto_record_meetings else "false"}',
        "",
        "[transcription]",
        f'formats = [{", ".join(f"\"{f}\"" for f in settings.transcription_formats)}]',
        f'output_dir = "{settings.transcription_output_dir}"',
        "",
        "[tts]",
        f'language = "{settings.tts_language}"',
        "",
        "[server]",
        f'host = "{settings.server_host}"',
        f"port = {settings.server_port}",
        "",
    ]

    CONFIG_FILE.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Settings saved to %s", CONFIG_FILE)


def _validate_formats(formats: list[str]) -> list[str]:
    valid = [f for f in formats if f in VALID_FORMATS]
    return valid if valid else ["txt"]
