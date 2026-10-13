import importlib
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from click.testing import CliRunner


class RecordIntegrationTests(unittest.TestCase):
    def run_recording(self, root, fail=False):
        class Recorder:
            def __init__(self, queue, **kwargs):
                self.queue = queue
            def start(self):
                for start in (0, 10):
                    self.queue.put(types.SimpleNamespace(audio=np.array([start]), start_time=start, duration=1))
                self.queue.put(None)
            def stop(self):
                pass
        def transcribe(audio, model):
            if fail and audio[0] == 10:
                raise RuntimeError('decoder failed')
            return {'text':'Oferta CRM', 'segments':[{'start':0,'end':1,'text':'Oferta CRM'}], 'language':'pl'}
        recorder = types.ModuleType('whisper_daemon.meeting_recorder')
        recorder.AudioChunk = object
        recorder.MeetingRecorder = Recorder
        transcriber = types.ModuleType('whisper_daemon.transcriber')
        transcriber.preload_model = lambda model: None
        transcriber.transcribe_full = transcribe
        with patch.dict('sys.modules', {'whisper_daemon.meeting_recorder':recorder,'whisper_daemon.transcriber':transcriber}):
            from whisper_daemon import __main__ as main
            with patch.object(main, "_setup_logging", lambda verbose: None):
                result = CliRunner().invoke(main.cli,['record',str(root/'spotkanie.txt'),'--orbit-ready'])
            self.assertEqual(result.exit_code,0,result.output+str(result.exception))
        return next(root.glob('recording_*'))

    def test_cli_waits_for_all_chunks_and_exports_final_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=self.run_recording(Path(tmp))
            manifest=json.loads((folder/'meeting.ready.json').read_text())
            transcript=json.loads((folder/'transcript.json').read_text())
            self.assertEqual(len(transcript['segments']),2)
            self.assertEqual(transcript['segments'][1]['start'],10)
            self.assertTrue(manifest['transcription_complete'])

    def test_cli_partial_success_never_signals_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=self.run_recording(Path(tmp),fail=True)
            self.assertFalse((folder/'meeting.ready.json').exists())
            self.assertTrue((folder/'meeting.failed.json').exists())
