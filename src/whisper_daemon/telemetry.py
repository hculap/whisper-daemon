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


# -- Meeting telemetry --

_meeting: dict = {}
_meeting_chunks: list[dict] = []


def meeting_start() -> None:
    """Start tracking a meeting recording session."""
    global _meeting, _meeting_chunks
    _meeting = {"type": "meeting", "start": time.time(), "start_mono": time.monotonic()}
    _meeting_chunks = []


def meeting_chunk_queued(chunk_num: int, audio_sec: float, start_time: float) -> None:
    """Record when a chunk is emitted from recorder."""
    _meeting_chunks.append({
        "chunk": chunk_num,
        "audio_sec": round(audio_sec, 1),
        "chunk_start_sec": round(start_time, 1),
        "queued_at": time.monotonic(),
    })


def meeting_chunk_transcribed(chunk_num: int, chars: int, segments: int) -> None:
    """Record when a chunk finishes transcription."""
    for c in _meeting_chunks:
        if c["chunk"] == chunk_num and "transcribed_at" not in c:
            c["transcribed_at"] = time.monotonic()
            c["chars"] = chars
            c["segments"] = segments
            c["transcribe_sec"] = round(c["transcribed_at"] - c["queued_at"], 3)
            break


def meeting_stop(chunks_total: int, output_dir: str) -> None:
    """Finalize and write meeting telemetry."""
    if not _meeting:
        return

    now = time.monotonic()
    _meeting["stop"] = time.time()
    _meeting["duration_total_s"] = round(now - _meeting["start_mono"], 1)
    _meeting["chunks_total"] = chunks_total
    _meeting["output_dir"] = output_dir
    _meeting["chunks"] = _meeting_chunks

    # Summary stats
    transcribe_times = [c["transcribe_sec"] for c in _meeting_chunks if "transcribe_sec" in c]
    if transcribe_times:
        _meeting["transcribe_avg_s"] = round(sum(transcribe_times) / len(transcribe_times), 3)
        _meeting["transcribe_max_s"] = round(max(transcribe_times), 3)

    audio_total = sum(c.get("audio_sec", 0) for c in _meeting_chunks)
    _meeting["audio_total_s"] = round(audio_total, 1)

    try:
        with open(TELEMETRY_FILE, "a") as f:
            f.write(json.dumps(_meeting, default=str) + "\n")
    except Exception:
        logger.debug("Failed to write meeting telemetry")
