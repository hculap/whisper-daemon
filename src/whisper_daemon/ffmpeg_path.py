"""Make ffmpeg discoverable regardless of how the daemon was launched.

File transcription decodes audio/video via ``mlx_whisper.audio.load_audio``,
which shells out to the ``ffmpeg`` binary. When the daemon runs under launchd
(the supervised menu-bar path), it inherits a minimal ``PATH`` —
``/usr/bin:/bin:/usr/sbin:/sbin`` — which excludes Homebrew and MacPorts, so
``ffmpeg`` is not found and every file transcription (m4a, mp4, mp3, …) fails
with FileNotFoundError. Running the CLI from a terminal works because the shell
PATH already contains the install prefix; this module closes that gap for the
launchd process.
"""

import logging
import os
import shutil

logger = logging.getLogger(__name__)

# launchd LaunchAgents get PATH=/usr/bin:/bin:/usr/sbin:/sbin. These are the
# standard install prefixes for ffmpeg on macOS that are NOT in that default:
# Homebrew on Apple Silicon, Homebrew on Intel, and MacPorts.
COMMON_BIN_DIRS = ("/opt/homebrew/bin", "/usr/local/bin", "/opt/local/bin")


def ensure_ffmpeg_on_path() -> str | None:
    """Return a resolvable path to ffmpeg, augmenting PATH if necessary.

    If ffmpeg is already discoverable, returns it unchanged. Otherwise appends
    the common macOS install prefixes to ``os.environ['PATH']`` (process-wide,
    so later subprocesses inherit it) and re-checks. Returns None if ffmpeg is
    genuinely not installed.
    """
    found = shutil.which("ffmpeg")
    if found:
        return found

    parts = [p for p in os.environ.get("PATH", "").split(os.pathsep) if p]
    changed = False
    for directory in COMMON_BIN_DIRS:
        if directory not in parts and os.path.isdir(directory):
            parts.append(directory)
            changed = True

    if changed:
        os.environ["PATH"] = os.pathsep.join(parts)
        found = shutil.which("ffmpeg")
        if found:
            logger.info("ffmpeg resolved after PATH augmentation: %s", found)

    return found
