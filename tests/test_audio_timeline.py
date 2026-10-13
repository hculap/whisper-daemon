import unittest
import numpy as np

class AudioTimelineTests(unittest.TestCase):
    def test_keeps_silence_offsets_and_mixes_overlapping_sources(self):
        from whisper_daemon.audio_timeline import mix_timeline
        chunks = [(0, np.array([1.,1.])), (1, np.array([-.5,-.5])), (3, np.array([.8]))]
        np.testing.assert_allclose(mix_timeline(chunks, sample_rate=1), [1.,.25,-.5,.8])
        np.testing.assert_allclose(mix_timeline([(2, np.array([.7]))], sample_rate=1), [0.,0.,.7])
