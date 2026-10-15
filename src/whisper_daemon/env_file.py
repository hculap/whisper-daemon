"""Read KEY=VALUE settings (e.g. HF_TOKEN) from the project's .env file.

Only model provisioning needs them; normal recording stays offline. Values are
returned, never written into os.environ, and never logged.
"""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Repository root of the editable install; a missing file simply yields nothing.
PROJECT_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


def read_env_file(path: Path) -> dict[str, str]:
    """Parse ``KEY=VALUE`` lines: optional ``export``, quoted values, ``#`` comments."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        logger.warning("Cannot read %s: %s", path, exc.strerror)
        return {}
    pairs = (_parse_line(line) for line in text.splitlines())
    return dict(pair for pair in pairs if pair)


def _parse_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped[len("export "):].lstrip()
    key, sep, value = stripped.partition("=")
    key = key.strip()
    if not sep or not key.isidentifier():
        return None
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    return key, value
