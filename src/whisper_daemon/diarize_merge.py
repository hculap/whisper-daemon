"""Merge speaker diarization segments with Whisper transcription results."""

from whisper_daemon.diarizer import SpeakerSegment

SAME_SPEAKER_GAP = 2.0  # seconds — merge consecutive same-speaker segments


def merge_speakers_with_transcript(
    speaker_segments: list[SpeakerSegment],
    transcript_result: dict,
) -> dict:
    """Assign speaker labels to transcript segments by timestamp overlap.

    For each Whisper segment, finds the speaker with maximum temporal
    overlap and assigns that speaker. Adds ``"speaker"`` to each segment
    and ``"speakers"`` to the top-level result.

    Returns a new dict (no mutation).
    """
    if not speaker_segments:
        return transcript_result

    segments = transcript_result.get("segments", [])
    labeled_segments = [
        {**seg, "speaker": _find_speaker(seg, speaker_segments)}
        for seg in segments
    ]

    all_speaker_ids = sorted({seg.speaker for seg in speaker_segments})
    speakers = [
        {"id": sid, "label": f"Speaker {sid + 1}"}
        for sid in all_speaker_ids
    ]

    return {
        **transcript_result,
        "segments": labeled_segments,
        "speakers": speakers,
    }


def _find_speaker(
    transcript_seg: dict,
    speaker_segments: list[SpeakerSegment],
) -> int:
    """Find speaker with maximum overlap for a transcript segment."""
    seg_start = transcript_seg["start"]
    seg_end = transcript_seg["end"]

    best_speaker = 0
    best_overlap = 0.0

    for sp in speaker_segments:
        overlap_start = max(seg_start, sp.start)
        overlap_end = min(seg_end, sp.end)
        overlap = max(0.0, overlap_end - overlap_start)

        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = sp.speaker

    return best_speaker
