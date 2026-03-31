"""Output format converters for transcription results."""

import json

COMPACT_STRIP_KEYS = {"tokens", "seek", "compression_ratio", "temperature", "avg_logprob", "no_speech_prob"}


def to_txt(result: dict) -> str:
    return result.get("text", "").strip()


def to_srt(result: dict) -> str:
    segments = result.get("segments", [])
    lines: list[str] = []

    for i, seg in enumerate(segments, start=1):
        start = _format_timestamp_srt(seg["start"])
        end = _format_timestamp_srt(seg["end"])
        text = seg["text"].strip()
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")

    return "\n".join(lines)


def to_vtt(result: dict) -> str:
    segments = result.get("segments", [])
    lines: list[str] = ["WEBVTT", ""]

    for seg in segments:
        start = _format_timestamp_vtt(seg["start"])
        end = _format_timestamp_vtt(seg["end"])
        text = seg["text"].strip()
        lines.append(f"{start} --> {end}\n{text}\n")

    return "\n".join(lines)


def to_json(result: dict) -> str:
    cleaned = _strip_segments(result)
    return json.dumps(cleaned, ensure_ascii=False, indent=2)


def _strip_segments(result: dict) -> dict:
    """Strip verbose keys (tokens, seek, etc.) from segments for compact output."""
    segments = result.get("segments", [])
    compact_segments = [
        {
            "id": seg.get("id"),
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip(),
        }
        for seg in segments
    ]
    return {
        "text": result.get("text", "").strip(),
        "segments": compact_segments,
        "language": result.get("language", ""),
    }


FORMATTERS = {
    "txt": to_txt,
    "srt": to_srt,
    "vtt": to_vtt,
    "json": to_json,
}


def _format_timestamp_srt(seconds: float) -> str:
    """Format seconds as HH:MM:SS,mmm (SRT format)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _format_timestamp_vtt(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm (WebVTT format)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
