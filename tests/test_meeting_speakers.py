import hashlib
import json
from datetime import datetime, timezone

import pytest

from whisper_daemon.diarizer import SpeakerSegment
from whisper_daemon.meeting_manifest import begin_meeting, finalize_meeting
from whisper_daemon.meeting_speakers import add_speakers, label_meeting

ONLINE = {
    "text": "Dzień dobry. Cześć. Do jutra. Pa.",
    "segments": [
        {"start": 0.0, "end": 2.0, "text": "Dzień dobry.", "source": "browser"},
        {"start": 3.0, "end": 5.0, "text": "Cześć.", "source": "mic"},
        {"start": 6.0, "end": 8.0, "text": "Do jutra.", "source": "browser"},
        {"start": 9.0, "end": 10.0, "text": "Pa.", "source": "browser"},
    ],
}
IN_ROOM = {
    "text": "Zaczynamy. Dobrze.",
    "segments": [
        {"start": 0.0, "end": 2.0, "text": "Zaczynamy.", "source": "mic"},
        {"start": 3.0, "end": 5.0, "text": "Dobrze.", "source": "mic"},
    ],
}


class FakeDiarizer:
    """Participants 7 then 3 on the diarized track (raw pyannote ids)."""

    def __init__(self, segments=None):
        self.paths = []
        self.segments = segments if segments is not None else [
            SpeakerSegment(speaker=7, start=0.0, end=2.5), SpeakerSegment(speaker=3, start=5.5, end=8.5)]

    def __call__(self, path, num_speakers=None):
        self.paths.append(path.rsplit("/", 1)[-1])
        return self.segments


def _tracks(root):
    for name in ("recording.wav", "mic.wav", "browser.wav"):
        (root / name).write_bytes(b"audio")


def _ready_meeting(root, transcript=ONLINE):
    identity = begin_meeting(root, "Daily", started_at=datetime(2026, 10, 15, 7, 30, tzinfo=timezone.utc),
                             timezone_name="Europe/Warsaw")
    _tracks(root)
    (root / "transcript.txt").write_text("unlabelled", encoding="utf-8")
    return finalize_meeting(root, identity, transcript, [])


def test_online_meeting_mic_is_you_and_only_participants_are_diarized(tmp_path):
    _tracks(tmp_path)
    diarize = FakeDiarizer()
    labelled = label_meeting(tmp_path, ONLINE, diarize)
    assert diarize.paths == ["browser.wav"]
    assert labelled["speakers"] == [{"id": 0, "label": "Ty"}, {"id": 1, "label": "Uczestnik 1"},
                                    {"id": 2, "label": "Uczestnik 2"}]
    # Numbered by first appearance; the last line has no overlap and goes to the nearest speaker.
    assert [s["speaker"] for s in labelled["segments"]] == [1, 0, 2, 2]
    assert ONLINE["segments"][0].get("speaker") is None  # input untouched


def test_online_meeting_without_the_model_still_separates_you_from_participants(tmp_path):
    labelled = label_meeting(tmp_path, ONLINE, None)
    assert labelled["speakers"] == [{"id": 0, "label": "Ty"}, {"id": 1, "label": "Uczestnicy"}]
    assert [s["speaker"] for s in labelled["segments"]] == [1, 0, 1, 1]


def test_microphone_only_recording_diarizes_the_whole_recording(tmp_path):
    _tracks(tmp_path)
    diarize = FakeDiarizer([SpeakerSegment(speaker=0, start=0.0, end=2.5),
                            SpeakerSegment(speaker=1, start=2.5, end=6.0)])
    labelled = label_meeting(tmp_path, IN_ROOM, diarize)
    assert diarize.paths == ["recording.wav"]
    assert [s["speaker"] for s in labelled["segments"]] == [0, 1]
    assert label_meeting(tmp_path, IN_ROOM, None) == IN_ROOM


def test_add_speakers_rewrites_exports_and_publishes_the_next_revision(tmp_path):
    first = _ready_meeting(tmp_path)
    result = add_speakers(tmp_path, diarize=FakeDiarizer())

    transcript = json.loads((tmp_path / "transcript.json").read_text())
    assert [s["label"] for s in transcript["speakers"]] == ["Ty", "Uczestnik 1", "Uczestnik 2"]
    text = (tmp_path / "transcript.txt").read_text()
    assert "Ty" in text and "Uczestnik 2" in text
    assert not (tmp_path / "transcript.srt").exists()  # only exports the meeting already had
    assert json.loads((tmp_path / "transcript_original.json").read_text()) == ONLINE

    manifest = json.loads((tmp_path / "meeting.ready.json").read_text())
    assert result == {"speakers": 3, "revision": 2}
    assert manifest["revision"] == first["revision"] + 1
    assert manifest["meeting_id"] == first["meeting_id"]
    for entry in manifest["files"]:
        assert entry["sha256"] == hashlib.sha256((tmp_path / entry["path"]).read_bytes()).hexdigest()


def test_labelled_transcripts_are_kept_unless_forced(tmp_path):
    _ready_meeting(tmp_path)
    add_speakers(tmp_path, diarize=FakeDiarizer())
    with pytest.raises(ValueError, match="already has speaker labels"):
        add_speakers(tmp_path, diarize=FakeDiarizer())
    assert add_speakers(tmp_path, diarize=FakeDiarizer(), force=True)["revision"] == 2  # nothing changed


def test_no_speech_changes_nothing(tmp_path):
    _ready_meeting(tmp_path, IN_ROOM)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    with pytest.raises(RuntimeError, match="no speech"):
        add_speakers(tmp_path, diarize=FakeDiarizer([]))
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()} == before


def test_a_meeting_never_published_gets_labels_without_a_manifest(tmp_path):
    _tracks(tmp_path)
    (tmp_path / "transcript.json").write_text(json.dumps(ONLINE), encoding="utf-8")
    assert add_speakers(tmp_path, diarize=FakeDiarizer()) == {"speakers": 3, "revision": None}
    assert not (tmp_path / "meeting.ready.json").exists()


def test_cli_explains_how_to_prepare_a_missing_model(monkeypatch, tmp_path):
    from click.testing import CliRunner
    from whisper_daemon import __main__ as cli
    monkeypatch.setattr(cli, "_speaker_model_ready", lambda requested: False)
    result = CliRunner().invoke(cli.cli, ["speakers", str(tmp_path)])
    assert result.exit_code != 0
    assert "models prepare --diarization" in result.output
