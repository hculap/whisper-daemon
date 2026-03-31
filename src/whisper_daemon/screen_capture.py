"""Periodic screenshot capture with perceptual change detection (dHash).

Supports multiple displays — each captured separately with independent
change detection. Files named: {timestamp}_display{N}.png
"""

import logging
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image
from Quartz import CGGetActiveDisplayList

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL = 5.0  # seconds between capture attempts
DHASH_SIZE = 16  # 16x16 grid = 256-bit hash
CHANGE_THRESHOLD = 0.12  # hamming distance — ignores cursor/clock, catches slide changes


class ScreenCapture:
    """Captures screenshots of all displays at regular intervals.

    Each display is tracked independently for change detection.
    Only saves when screen content meaningfully changes.
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
        self._last_hashes: dict[int, np.ndarray] = {}  # display_id → last hash
        self._saved_count = 0

    @property
    def saved_count(self) -> int:
        return self._saved_count

    def start(self) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._running = True
        self._start_time = time.monotonic()
        self._last_hashes = {}
        self._saved_count = 0
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        display_count = _get_display_count()
        logger.info(
            "Screen capture started (displays=%d, interval=%.1fs, threshold=%.2f)",
            display_count, self._interval, self._threshold,
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
                self._capture_all_displays()
            except Exception:
                logger.exception("Screenshot capture failed")
            time.sleep(self._interval)

    def _capture_all_displays(self) -> None:
        elapsed = time.monotonic() - self._start_time
        timestamp_sec = int(elapsed)

        display_ids = _get_display_ids()

        for i, display_id in enumerate(display_ids, start=1):
            self._capture_display(display_id, i, timestamp_sec)

    def _capture_display(self, display_id: int, display_num: int, timestamp_sec: int) -> None:
        temp_path = self._output_dir / f"_temp_d{display_num}.png"

        result = subprocess.run(
            ["screencapture", "-x", "-C", "-D", str(display_id), str(temp_path)],
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

        last_hash = self._last_hashes.get(display_id)
        if last_hash is not None:
            distance = _hamming_distance(last_hash, current_hash)
            if distance < self._threshold:
                temp_path.unlink(missing_ok=True)
                return

        final_path = self._output_dir / f"{timestamp_sec:06d}_display{display_num}.png"
        temp_path.rename(final_path)
        self._last_hashes[display_id] = current_hash
        self._saved_count += 1
        logger.debug("Screenshot saved: %s", final_path.name)


def _get_display_ids() -> list[int]:
    """Get list of active display IDs."""
    err, display_ids, count = CGGetActiveDisplayList(10, None, None)
    if err != 0 or not display_ids:
        return [1]  # fallback to main display
    return list(display_ids[:count])


def _get_display_count() -> int:
    return len(_get_display_ids())


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
