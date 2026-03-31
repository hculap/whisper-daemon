"""Periodic screenshot capture with perceptual change detection (dHash)."""

import logging
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL = 5.0  # seconds between capture attempts
DHASH_SIZE = 16  # 16x16 grid = 256-bit hash
CHANGE_THRESHOLD = 0.12  # hamming distance — ignores cursor/clock, catches slide changes


class ScreenCapture:
    """Captures screenshots at regular intervals, skipping unchanged frames.

    Uses dHash (difference hash) for perceptual change detection.
    Only saves when screen content meaningfully changes — ignores cursor
    blinks, clock updates, and notification badges.
    """

    def __init__(
        self,
        output_dir: Path,
        interval: float = DEFAULT_INTERVAL,
        threshold: float = CHANGE_THRESHOLD,
    ) -> None:
        self._output_dir = output_dir / "screenshots"
        self._interval = interval
        self._threshold = threshold
        self._running = False
        self._thread: threading.Thread | None = None
        self._start_time: float = 0.0
        self._last_hash: np.ndarray | None = None
        self._saved_count = 0

    @property
    def saved_count(self) -> int:
        return self._saved_count

    def start(self) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._running = True
        self._start_time = time.monotonic()
        self._last_hash = None
        self._saved_count = 0
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info(
            "Screen capture started (interval=%.1fs, dHash threshold=%.2f)",
            self._interval, self._threshold,
        )

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("Screen capture stopped — %d screenshots saved", self._saved_count)

    def _capture_loop(self) -> None:
        while self._running:
            try:
                self._capture_frame()
            except Exception:
                logger.exception("Screenshot capture failed")
            time.sleep(self._interval)

    def _capture_frame(self) -> None:
        elapsed = time.monotonic() - self._start_time
        timestamp_sec = int(elapsed)

        temp_path = self._output_dir / "_temp.png"
        result = subprocess.run(
            ["screencapture", "-x", "-C", str(temp_path)],
            capture_output=True,
            timeout=5,
        )
        if result.returncode != 0 or not temp_path.exists():
            return

        try:
            img = Image.open(temp_path)
        except Exception:
            temp_path.unlink(missing_ok=True)
            return

        current_hash = _dhash(img)

        if self._last_hash is not None:
            distance = _hamming_distance(self._last_hash, current_hash)
            if distance < self._threshold:
                temp_path.unlink(missing_ok=True)
                return

        final_path = self._output_dir / f"{timestamp_sec:06d}.png"
        temp_path.rename(final_path)
        self._last_hash = current_hash
        self._saved_count += 1
        logger.debug("Screenshot saved: %s (at %ds)", final_path.name, timestamp_sec)


def _dhash(image: Image.Image, hash_size: int = DHASH_SIZE) -> np.ndarray:
    """Compute difference hash — compare each pixel to its right neighbor."""
    gray = image.convert("L").resize(
        (hash_size + 1, hash_size), Image.LANCZOS
    )
    pixels = np.asarray(gray)
    return (pixels[:, 1:] > pixels[:, :-1]).flatten()


def _hamming_distance(hash1: np.ndarray, hash2: np.ndarray) -> float:
    """Normalized hamming distance. 0.0 = identical, 1.0 = completely different."""
    return np.count_nonzero(hash1 != hash2) / len(hash1)
