import concurrent.futures
import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path


class ManifestTests(unittest.TestCase):
    def test_ready_is_last_and_matches_all_files(self):
        from whisper_daemon.meeting_manifest import begin_meeting, finalize_meeting
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            identity = begin_meeting(root, 'CRM', started_at=datetime(2026,10,13,14,tzinfo=timezone.utc), timezone_name='Europe/Warsaw')
            self.assertFalse((root / 'meeting.ready.json').exists())
            (root / 'recording.wav').write_bytes(b'audio')
            (root / 'screenshots').mkdir()
            (root / 'screenshots/1.jpg').write_bytes(b'image')
            (root / 'transcript_live.txt').write_text('partial')
            result = {'text': 'Oferta', 'segments': [{'start':0, 'end':2, 'text':'Oferta'}]}
            manifest = finalize_meeting(root, identity, result, [])
            self.assertEqual(manifest['meeting_id'], identity['meeting_id'])
            self.assertEqual(manifest['revision'], 1)
            self.assertEqual({f['path'] for f in manifest['files']}, {'recording.wav','transcript.json','screenshots/1.jpg'})
            for file in manifest['files']:
                content = (root / file['path']).read_bytes()
                self.assertEqual(file['sha256'], hashlib.sha256(content).hexdigest())
                self.assertEqual(file['size'], len(content))
            self.assertEqual(json.loads((root / 'meeting.ready.json').read_text()), manifest)
            with self.assertRaises(FileExistsError):
                begin_meeting(root, 'second recording')

    def test_chunk_failure_writes_failed_status_and_never_ready(self):
        from whisper_daemon.meeting_manifest import begin_meeting, finalize_meeting, collect_futures
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            identity = begin_meeting(root, 'CRM')
            good, bad = concurrent.futures.Future(), concurrent.futures.Future()
            good.set_result({'text':'ok', 'segments':[]})
            bad.set_exception(RuntimeError('decoder failed'))
            futures, results, errors = {good:0,bad:30}, [], []
            self.assertTrue(collect_futures(futures, results, errors))
            self.assertEqual(len(results), 1)
            self.assertEqual(len(errors), 1)
            self.assertIsNone(finalize_meeting(root, identity, {'text':'ok','segments':[]}, errors))
            self.assertFalse((root / 'meeting.ready.json').exists())
            self.assertTrue((root / 'meeting.failed.json').exists())

    def test_symlink_and_empty_recording_cannot_be_ready(self):
        from whisper_daemon.meeting_manifest import begin_meeting, finalize_meeting
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'meeting'; root.mkdir()
            identity = begin_meeting(root,'test')
            self.assertIsNone(finalize_meeting(root,identity,{'text':'','segments':[]},[]))
            (root / 'escape').symlink_to(Path(tmp))
            with self.assertRaises(ValueError):
                finalize_meeting(root,identity,{'text':'yes','segments':[]},[])
            self.assertFalse((root / 'meeting.ready.json').exists())


    def test_pipeline_transcript_is_published_without_recovery_internals(self):
        from whisper_daemon.meeting_manifest import begin_meeting, finalize_meeting
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            identity = begin_meeting(root, 'CRM')
            transcript = json.dumps({'text': 'Oferta', 'segments': [{'start': 0, 'end': 2, 'text': 'Oferta'}]})
            (root / 'transcript.json').write_text(transcript)
            for name in ('browser.f32', 'recording.wav', 'session.json', 'chunks/1.json'):
                (root / name).parent.mkdir(exist_ok=True)
                (root / name).write_bytes(b'x')
            manifest = finalize_meeting(root, identity, None, [])
            self.assertEqual({f['path'] for f in manifest['files']}, {'transcript.json', 'recording.wav', 'session.json'})
            self.assertEqual((root / 'transcript.json').read_text(), transcript)  # not rewritten

    def test_unacknowledged_chunk_job_blocks_ready(self):
        from whisper_daemon.meeting_manifest import begin_meeting, finalize_meeting
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            identity = begin_meeting(root, 'CRM')
            (root / 'chunks').mkdir()
            (root / 'chunks/0001.npz').write_bytes(b'pending audio')
            self.assertIsNone(finalize_meeting(root, identity, {'text': 'ok', 'segments': []}, []))
            failed = json.loads((root / 'meeting.failed.json').read_text())
            self.assertEqual(failed['transcription_errors'], ['1 transcription job(s) never completed'])
            self.assertFalse((root / 'meeting.ready.json').exists())


class TargetTests(unittest.TestCase):
    """A meeting captured locally is published to a separate (synced) folder."""
    result = {'text': 'Oferta', 'segments': [{'start': 0, 'end': 2, 'text': 'Oferta'}]}

    def capture(self, tmp):
        from whisper_daemon.meeting_manifest import begin_meeting
        capture, target = Path(tmp) / 'capture' / 'recording_x', Path(tmp) / 'Sync' / 'recording_x'
        identity = begin_meeting(capture, 'CRM')
        for name in ('recording.wav', 'mic.f32', 'chunks/1.json', 'screenshots/1.jpg', 'transcript_live.txt'):
            (capture / name).parent.mkdir(exist_ok=True)
            (capture / name).write_bytes(name.encode())
        return capture, target, identity

    def test_only_published_files_reach_the_target_and_the_capture_is_removed(self):
        from whisper_daemon.meeting_manifest import finalize_meeting
        with tempfile.TemporaryDirectory() as tmp:
            capture, target, identity = self.capture(tmp)
            manifest = finalize_meeting(capture, identity, self.result, [], target=target)
            self.assertFalse(capture.exists())
            self.assertEqual(sorted(p.relative_to(target).as_posix() for p in target.rglob('*') if p.is_file()),
                             ['meeting.ready.json', 'recording.wav', 'screenshots/1.jpg', 'transcript.json'])
            for file in manifest['files']:
                self.assertEqual(file['sha256'], hashlib.sha256((target / file['path']).read_bytes()).hexdigest())
            self.assertEqual(json.loads((target / 'meeting.ready.json').read_text()), manifest)

    def test_a_failed_meeting_stays_in_the_capture(self):
        from whisper_daemon.meeting_manifest import finalize_meeting
        with tempfile.TemporaryDirectory() as tmp:
            capture, target, identity = self.capture(tmp)
            self.assertIsNone(finalize_meeting(capture, identity, {'text': '', 'segments': []}, [], target=target))
            self.assertFalse(target.exists())
            self.assertTrue((capture / 'meeting.failed.json').exists() and (capture / 'mic.f32').exists())

    def test_an_existing_target_is_never_merged_into(self):
        from whisper_daemon.meeting_manifest import finalize_meeting
        with tempfile.TemporaryDirectory() as tmp:
            capture, target, identity = self.capture(tmp)
            target.mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                finalize_meeting(capture, identity, self.result, [], target=target)
            self.assertEqual(list(target.iterdir()), [])
            self.assertTrue((capture / 'mic.f32').exists())

    def test_a_partial_copy_is_removed_and_the_capture_kept(self):
        from unittest import mock
        from whisper_daemon import meeting_manifest
        with tempfile.TemporaryDirectory() as tmp:
            capture, target, identity = self.capture(tmp)
            with mock.patch.object(meeting_manifest.shutil, 'copyfile', side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    meeting_manifest.finalize_meeting(capture, identity, self.result, [], target=target)
            self.assertFalse(target.exists())
            self.assertTrue((capture / 'recording.wav').exists() and (capture / 'mic.f32').exists())


if __name__ == '__main__': unittest.main()
