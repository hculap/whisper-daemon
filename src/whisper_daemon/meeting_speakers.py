"""Speaker labels for meeting transcripts.

An online meeting records each side on its own track, so the sources decide
who is who: everything from the microphone is the user ("Ty"), and only the
participants' track (browser.wav) is diarized into "Uczestnik 1", "Uczestnik 2"…
A microphone-only recording (e.g. a meeting in a room) has no such split, so
the whole recording is diarized.

The live pipeline labels a meeting when it is saved; ``add_speakers`` does the
same later for a meeting saved before the speaker model was ready, rewrites
its exports and republishes it for Orbit.
"""
import json
import logging
import shutil
from pathlib import Path
from typing import Callable

from whisper_daemon.formats import FORMATTERS
from whisper_daemon.meeting_manifest import publish_revision
from whisper_daemon.recording_store import atomic_json

logger = logging.getLogger(__name__)

MIC_SPEAKER = 0
MIC_LABEL = "Ty"
PARTICIPANT_LABEL = "Uczestnik {}"
PARTICIPANTS_LABEL = "Uczestnicy"  # participants' speech that could not be split
# transcript.json is the machine-readable copy; these are its readable exports.
TEXT_EXPORTS = ("txt", "srt", "vtt")


def label_meeting(folder: Path, transcript: dict, diarize: Callable | None,
                  *, num_speakers: int | None = None) -> dict:
    """Return the transcript with speaker labels (a new dict).

    ``diarize`` is ``diarizer.diarize_file`` or None when the speaker model is
    unavailable; an online meeting is then still split into user/participants.
    ``num_speakers`` is the expected count on the diarized track: participants
    for an online meeting, everyone for a microphone-only recording.
    """
    segments = transcript.get("segments", [])
    if not any(seg.get("source") == "browser" for seg in segments):
        if diarize is None:
            return transcript
        from whisper_daemon.diarize_merge import merge_speakers_with_transcript
        found = diarize(str(Path(folder) / "recording.wav"), num_speakers=num_speakers)
        return merge_speakers_with_transcript(found, transcript) if found else transcript
    browser = Path(folder) / "browser.wav"
    found = diarize(str(browser), num_speakers=num_speakers) if diarize and browser.exists() else []
    return label_by_source(transcript, found)


def label_by_source(transcript: dict, participant_segments: list) -> dict:
    """Microphone segments are the user; participants come from their own track.

    Participants are numbered in the order they first speak in the transcript.
    Without diarized participant segments they share one "Uczestnicy" label.
    """
    raw = [
        MIC_SPEAKER if seg.get("source") != "browser"
        else ("p", _closest_speaker(seg, participant_segments) if participant_segments else None)
        for seg in transcript.get("segments", [])
    ]
    order = list(dict.fromkeys(key for key in raw if key != MIC_SPEAKER))
    number = {key: index for index, key in enumerate(order, start=1)}
    labelled = [{**seg, "speaker": number.get(key, MIC_SPEAKER)}
                for seg, key in zip(transcript.get("segments", []), raw)]
    speakers = []
    if MIC_SPEAKER in raw:
        speakers.append({"id": MIC_SPEAKER, "label": MIC_LABEL})
    for key, index in number.items():
        label = PARTICIPANTS_LABEL if key[1] is None else PARTICIPANT_LABEL.format(index)
        speakers.append({"id": index, "label": label})
    return {**transcript, "segments": labelled, "speakers": speakers}


def _closest_speaker(segment: dict, speaker_segments: list) -> int:
    """Speaker with the most overlap, else the one speaking nearest in time."""
    start, end = segment["start"], segment["end"]

    def rank(sp):
        overlap = min(end, sp.end) - max(start, sp.start)
        return (overlap, 0.0) if overlap > 0 else (0.0, overlap)

    return max(speaker_segments, key=rank).speaker


def add_speakers(folder: Path, *, diarize: Callable | None = None,
                 num_speakers: int | None = None, force: bool = False) -> dict:
    """Label a saved meeting's transcript and republish the meeting.

    Keeps the transcript it replaces as transcript_original.json (as review
    edits do), rewrites only the exports the meeting already has, and returns
    ``{"speakers": count, "revision": new revision or None if never published}``.
    """
    path = Path(folder)
    transcript_path = path / "transcript.json"
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    if not transcript.get("segments"):
        raise ValueError("The transcript is empty")
    if transcript.get("speakers") and not force:
        raise ValueError("The transcript already has speaker labels; use --force to replace them")
    if diarize is None:
        from whisper_daemon.diarizer import diarize_file as diarize
    labelled = label_meeting(path, transcript, diarize, num_speakers=num_speakers)
    if not labelled.get("speakers"):
        raise RuntimeError("Speaker recognition found no speech in recording.wav")

    original = path / "transcript_original.json"
    if not original.exists():
        shutil.copy2(transcript_path, original)
    atomic_json(transcript_path, labelled)
    for fmt in TEXT_EXPORTS:
        target = path / f"transcript.{fmt}"
        if target.exists():
            temp = target.with_suffix(target.suffix + ".tmp")
            temp.write_text(FORMATTERS[fmt](labelled), encoding="utf-8")
            temp.replace(target)

    manifest = publish_revision(path)
    revision = manifest["revision"] if manifest else None
    logger.info("Labelled %d speakers in %s (revision %s)", len(labelled["speakers"]), path.name, revision)
    return {"speakers": len(labelled["speakers"]), "revision": revision}
