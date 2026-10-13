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


if __name__ == '__main__': unittest.main()
