"""Internal telemetry for latency analysis. Writes structured timing data to a JSON lines file."""

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

TELEMETRY_FILE = Path.home() / ".config" / "whisper-daemon" / "telemetry.jsonl"

_session_start: float = 0.0
_current: dict = {}


def start_session() -> None:
    """Call once at daemon startup."""
    global _session_start
    _session_start = time.monotonic()
    TELEMETRY_FILE.parent.mkdir(parents=True, exist_ok=True)


def mark(event: str, **extra: object) -> None:
    """Record a timestamped event in the current recording cycle."""
    global _current
    now = time.monotonic()
    ts = now - _session_start

    if event == "record_start":
        _current = {"record_start": ts}
    elif _current:
        _current[event] = ts

    for k, v in extra.items():
        _current[f"{event}_{k}"] = v


def flush() -> None:
    """Write the current cycle's timing data and compute deltas."""
    global _current
    if not _current:
        return

    rec = dict(_current)

    # Compute deltas
    t = rec.get
    if t("record_start") is not None and t("record_stop") is not None:
        rec["duration_recording_s"] = round(t("record_stop") - t("record_start"), 3)
    if t("record_stop") is not None and t("vad_silence") is not None:
        rec["duration_vad_wait_s"] = round(t("record_stop") - t("vad_silence"), 3)
    if t("record_stop") is not None and t("transcribe_start") is not None:
        rec["duration_stop_to_transcribe_s"] = round(t("transcribe_start") - t("record_stop"), 3)
    if t("transcribe_start") is not None and t("transcribe_done") is not None:
        rec["duration_transcribe_s"] = round(t("transcribe_done") - t("transcribe_start"), 3)
    if t("transcribe_done") is not None and t("paste_done") is not None:
        rec["duration_paste_s"] = round(t("paste_done") - t("transcribe_done"), 3)

    # End-to-end: from speech end (vad_silence or record_stop) to paste
    speech_end = t("vad_silence") or t("record_stop")
    if speech_end is not None and t("paste_done") is not None:
        rec["duration_end_to_end_s"] = round(t("paste_done") - speech_end, 3)

    # Preview stats
    if t("preview_start") is not None and t("preview_done") is not None:
        rec["duration_preview_s"] = round(t("preview_done") - t("preview_start"), 3)

    try:
        with open(TELEMETRY_FILE, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
    except Exception:
        logger.debug("Failed to write telemetry")

    _current = {}
