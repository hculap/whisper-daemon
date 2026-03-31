"""Periodic screenshot capture with change detection."""

import logging
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL = 5.0  # seconds between capture attempts
CHANGE_THRESHOLD = 0.02  # 2% pixel change required to save


class ScreenCapture:
    """Captures screenshots at regular intervals, skipping unchanged frames."""

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
        self._last_image: np.ndarray | None = None
        self._saved_count = 0

    @property
    def saved_count(self) -> int:
        return self._saved_count

    def start(self) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._running = True
        self._start_time = time.monotonic()
        self._last_image = None
        self._saved_count = 0
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info("Screen capture started (interval=%.1fs, threshold=%.0f%%)",
                     self._interval, self._threshold * 100)

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

        # Capture to temp file using macOS screencapture
        temp_path = self._output_dir / "_temp.png"
        result = subprocess.run(
            ["screencapture", "-x", "-C", str(temp_path)],
            capture_output=True,
            timeout=5,
        )
        if result.returncode != 0 or not temp_path.exists():
            return

        current_image = _load_image_as_array(temp_path)
        if current_image is None:
            temp_path.unlink(missing_ok=True)
            return

        if self._last_image is not None and not _has_changed(
            self._last_image, current_image, self._threshold
        ):
            temp_path.unlink(missing_ok=True)
            return

        # Screen changed — save with timestamp name
        final_path = self._output_dir / f"{timestamp_sec:06d}.png"
        temp_path.rename(final_path)
        self._last_image = current_image
        self._saved_count += 1
        logger.debug("Screenshot saved: %s (at %ds)", final_path.name, timestamp_sec)


def _load_image_as_array(path: Path) -> np.ndarray | None:
    """Load PNG as a small grayscale array for fast comparison."""
    try:
        # Use sips to get raw pixel data quickly (macOS built-in)
        # Resize to small thumbnail for fast comparison
        result = subprocess.run(
            [
                "sips",
                "-z", "64", "64",
                "-s", "format", "jpeg",
                str(path),
                "--out", str(path.with_suffix(".thumb.jpg")),
            ],
            capture_output=True,
            timeout=5,
        )
        thumb_path = path.with_suffix(".thumb.jpg")
        if result.returncode != 0 or not thumb_path.exists():
            return None

        data = np.frombuffer(thumb_path.read_bytes(), dtype=np.uint8)
        thumb_path.unlink(missing_ok=True)
        return data
    except Exception:
        return None


def _has_changed(prev: np.ndarray, curr: np.ndarray, threshold: float) -> bool:
    """Check if two image arrays differ by more than threshold."""
    if prev.shape != curr.shape:
        return True

    min_len = min(len(prev), len(curr))
    diff = np.abs(prev[:min_len].astype(np.int16) - curr[:min_len].astype(np.int16))
    changed_pixels = np.sum(diff > 10) / min_len
    return changed_pixels > threshold
