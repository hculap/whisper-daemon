import json
import tempfile
import types
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np
from click.testing import CliRunner


@dataclass(frozen=True)
class AudioChunk:  # meeting_recorder.AudioChunk; that module needs PortAudio
    audio: np.ndarray
    start_time: float
    duration: float
    source: str = 'mic'
    job_id: str = ''


class RecordIntegrationTests(unittest.TestCase):
    def run_recording(self, root, fail=False, orbit=True, storage_error=None):
        calls = []

        class Recorder:
            def __init__(self, queue, **kwargs):
                self.queue = queue
            def start(self):
                for start in (0, 10):
                    self.queue.put(AudioChunk(np.array([start], dtype=np.float32), start, 1))
            @property
            def error(self):
                if len(calls) == 2:  # Ctrl+C once both chunks reached Whisper
                    raise KeyboardInterrupt
                return storage_error
            def stop(self):
                pass
            def abort(self):
                pass

        def transcribe(audio, model, language=None):
            calls.append(audio[0])
            if fail and audio[0] == 10:
                raise RuntimeError('decoder failed')
            return {'text':'Oferta CRM', 'segments':[{'start':0,'end':1,'text':'Oferta CRM'}], 'language':'pl'}
        recorder = types.ModuleType('whisper_daemon.meeting_recorder')
        recorder.AudioChunk = AudioChunk
        recorder.MeetingRecorder = Recorder
        transcriber = types.ModuleType('whisper_daemon.transcriber')
        transcriber.preload_model = lambda model: None
        transcriber.transcribe_full = transcribe
        with patch.dict('sys.modules', {'whisper_daemon.meeting_recorder':recorder,'whisper_daemon.transcriber':transcriber}):
            from whisper_daemon import __main__ as main
            with patch.object(main, "_setup_logging", lambda verbose: None):
                args = ['record', str(root/'spotkanie.txt')] + (['--orbit-ready'] if orbit else [])
                result = CliRunner().invoke(main.cli, args)
            self.assertEqual(result.exit_code,1 if storage_error else 0,result.output+str(result.exception))
        return next(root.glob('recording_*' if orbit else 'spotkanie_audio_*'))

    def test_cli_waits_for_all_chunks_and_exports_final_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=self.run_recording(Path(tmp))
            manifest=json.loads((folder/'meeting.ready.json').read_text())
            transcript=json.loads((folder/'transcript.json').read_text())
            self.assertEqual(len(transcript['segments']),2)
            self.assertEqual(transcript['segments'][1]['start'],10)
            self.assertTrue(manifest['transcription_complete'])
            self.assertEqual(manifest['title'], 'spotkanie')
            self.assertFalse((folder/'meeting.in-progress.json').exists())
            # Everything Orbit needs lives in the one meeting folder; nothing beside it.
            self.assertFalse((Path(tmp)/'spotkanie.txt').exists())
            roles = {f['path']: f['role'] for f in manifest['files']}
            self.assertEqual(roles['transcript.json'], 'transcript')
            self.assertEqual(roles['recording.wav'], 'audio')
            self.assertIn('transcript.txt', roles)
            self.assertIn('session.json', roles)
            self.assertFalse([p for p in roles if p.startswith('chunks/') or p.endswith('.f32')])

    def test_cli_partial_success_never_signals_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=self.run_recording(Path(tmp),fail=True)
            self.assertFalse((folder/'meeting.ready.json').exists())
            failed = json.loads((folder/'meeting.failed.json').read_text())
            self.assertFalse(failed['transcription_complete'])
            self.assertIn('Chunk at 10.000s: RuntimeError', failed['transcription_errors'])
            # The failed job stays in the durable spool for recovery.
            self.assertIn('1 transcription job(s) never completed', failed['transcription_errors'])

    def test_cli_without_flag_keeps_the_plain_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=self.run_recording(Path(tmp), orbit=False)
            self.assertTrue((Path(tmp)/'spotkanie.txt').exists())
            self.assertFalse(list(folder.glob('meeting.*.json')))

    def test_cli_storage_failure_marks_meeting_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=self.run_recording(Path(tmp), storage_error=OSError('disk full'))
            self.assertFalse((folder/'meeting.ready.json').exists())
            failed = json.loads((folder/'meeting.failed.json').read_text())
            self.assertEqual(failed['transcription_errors'], ['RuntimeError: disk full'])
