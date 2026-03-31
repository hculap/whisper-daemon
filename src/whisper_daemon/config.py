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
    transcription_formats: list[str] = field(default_factory=lambda: ["txt"])
    transcription_output_dir: str = ""  # empty = same as input file

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

        return Settings(
            recording_dir=rec.get("dir", "~/Desktop"),
            recording_formats=_validate_formats(rec.get("formats", ["txt"])),
            recording_device=rec.get("device", ""),
            transcription_formats=_validate_formats(trans.get("formats", ["txt"])),
            transcription_output_dir=trans.get("output_dir", ""),
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
        "",
        "[transcription]",
        f'formats = [{", ".join(f"\"{f}\"" for f in settings.transcription_formats)}]',
        f'output_dir = "{settings.transcription_output_dir}"',
        "",
    ]

    CONFIG_FILE.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Settings saved to %s", CONFIG_FILE)


def _validate_formats(formats: list[str]) -> list[str]:
    valid = [f for f in formats if f in VALID_FORMATS]
    return valid if valid else ["txt"]
