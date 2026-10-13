"""Output format converters for transcription results.

All formatters handle optional speaker labels — if segments contain a
``"speaker"`` key, output includes speaker annotations. If absent,
output is unchanged (backward compatible).
"""

import json
import re

COMPACT_STRIP_KEYS = {"tokens", "seek", "compression_ratio", "temperature", "avg_logprob", "no_speech_prob"}


def to_txt(result: dict) -> str:
    segments = result.get("segments", [])
    if not segments:
        return result.get("text", "").strip()

    has_speakers = "speaker" in segments[0]
    if not has_speakers:
        return result.get("text", "").strip()

    speakers = {s["id"]: s["label"] for s in result.get("speakers", [])}
    paragraphs: list[str] = []
    current_speaker = -1
    current_texts: list[str] = []
    current_start = 0.0

    for seg in segments:
        speaker_id = seg.get("speaker", 0)
        if speaker_id != current_speaker:
            if current_texts:
                label = speakers.get(current_speaker, f"Speaker {current_speaker + 1}")
                timestamp = _format_timestamp_short(current_start)
                paragraphs.append(f"{label} [{timestamp}]\n{' '.join(current_texts)}")
            current_speaker = speaker_id
            current_texts = []
            current_start = seg["start"]
        current_texts.append(seg["text"].strip())

    if current_texts:
        label = speakers.get(current_speaker, f"Speaker {current_speaker + 1}")
        timestamp = _format_timestamp_short(current_start)
        paragraphs.append(f"{label} [{timestamp}]\n{' '.join(current_texts)}")

    return "\n\n".join(paragraphs)


def to_srt(result: dict) -> str:
    segments = result.get("segments", [])
    has_speakers = segments and "speaker" in segments[0]
    speakers = {s["id"]: s["label"] for s in result.get("speakers", [])} if has_speakers else {}
    lines: list[str] = []

    for i, seg in enumerate(segments, start=1):
        start = _format_timestamp_srt(seg["start"])
        end = _format_timestamp_srt(seg["end"])
        text = seg["text"].strip()
        if has_speakers:
            label = speakers.get(seg.get("speaker", 0), f"Speaker {seg.get('speaker', 0) + 1}")
            text = f"[{label}] {text}"
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")

    return "\n".join(lines)


def to_vtt(result: dict) -> str:
    segments = result.get("segments", [])
    has_speakers = segments and "speaker" in segments[0]
    speakers = {s["id"]: s["label"] for s in result.get("speakers", [])} if has_speakers else {}
    lines: list[str] = ["WEBVTT", ""]

    for seg in segments:
        start = _format_timestamp_vtt(seg["start"])
        end = _format_timestamp_vtt(seg["end"])
        text = seg["text"].strip()
        if has_speakers:
            label = speakers.get(seg.get("speaker", 0), f"Speaker {seg.get('speaker', 0) + 1}")
            text = f"<v {label}>{text}</v>"
        lines.append(f"{start} --> {end}\n{text}\n")

    return "\n".join(lines)


def to_json(result: dict) -> str:
    cleaned = _strip_segments(result)
    return json.dumps(cleaned, ensure_ascii=False, indent=2)


def _strip_segments(result: dict) -> dict:
    """Strip verbose keys (tokens, seek, etc.) from segments for compact output."""
    segments = result.get("segments", [])
    has_speakers = segments and "speaker" in segments[0]

    compact_segments = []
    for seg in segments:
        compact = {
            "id": seg.get("id"),
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip(),
        }
        if has_speakers:
            compact["speaker"] = seg["speaker"]
        for key in ("source", "words"):
            if key in seg:
                compact[key] = seg[key]
        compact_segments.append(compact)

    cleaned = {
        "text": result.get("text", "").strip(),
        "segments": compact_segments,
        "language": result.get("language", ""),
    }
    if "speakers" in result:
        cleaned["speakers"] = result["speakers"]
    return cleaned


FORMATTERS = {
    "txt": to_txt,
    "srt": to_srt,
    "vtt": to_vtt,
    "json": to_json,
}


def merge_chunk_results(results: list[tuple[float, dict]]) -> dict:
    """Merge on a shared clock, deduplicating within each source only.

    Word timestamps allow trimming just the overlap. Legacy segments without
    words use conservative suffix/prefix text matching; novel speech is never
    discarded merely because its segment starts inside an overlap.
    """
    sorted_results = sorted(results, key=lambda r: r[0])
    merged_segments: list[dict] = []
    high_water: dict[str, float] = {}
    previous_text: dict[str, str] = {}
    for start_offset, result in sorted_results:
        source = result.get("source", "mic")
        segments = result.get("segments", [])
        if not segments and result.get("text", "").strip():
            segments = [{"start": 0.0, "end": result.get("duration", 0.0),
                         "text": result["text"]}]
        for seg in segments:
            adjusted_start = seg["start"] + start_offset
            adjusted_end = seg["end"] + start_offset
            boundary = high_water.get(source, -1.0)
            text = seg.get("text", "").strip()
            words = [{**w, "start": w["start"] + start_offset,
                      "end": w["end"] + start_offset} for w in seg.get("words", [])]
            if adjusted_start < boundary:
                if words:
                    words = [w for w in words if w["end"] > boundary + 0.02]
                    text = " ".join(w["word"].strip() for w in words)
                    if words:
                        adjusted_start = max(boundary, words[0]["start"])
                else:
                    old = previous_text.get(source, "").split()
                    new = text.split()
                    norm = lambda t: re.sub(r"[^\w]", "", t.casefold())
                    for size in range(min(len(old), len(new)), 0, -1):
                        if list(map(norm, old[-size:])) == list(map(norm, new[:size])):
                            text = " ".join(new[size:])
                            break
            if not text:
                continue
            item = {**seg, "source": source, "start": adjusted_start,
                    "end": adjusted_end, "text": text}
            if words:
                item["words"] = words
            merged_segments.append(item)
            high_water[source] = max(boundary, adjusted_end)
            previous_text[source] = (previous_text.get(source, "") + " " + text)[-2000:]
    merged_segments.sort(key=lambda seg: (seg["start"], seg["end"], seg["source"]))
    for index, seg in enumerate(merged_segments):
        seg["id"] = index
    merged_text = " ".join(
        seg["text"].strip() for seg in merged_segments if seg.get("text", "").strip()
    )

    return {
        "text": merged_text,
        "segments": merged_segments,
        "language": sorted_results[0][1].get("language", "") if sorted_results else "",
    }


def _format_timestamp_short(seconds: float) -> str:
    """Format seconds as M:SS or H:MM:SS for human display."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


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
