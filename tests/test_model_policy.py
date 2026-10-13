import os
import subprocess
import sys
import pytest
from whisper_daemon.model_policy import enforce_offline


def test_offline_policy_overrides_opt_in_telemetry(monkeypatch):
    monkeypatch.delenv('WHISPER_PREPARE_MODELS', raising=False)
    monkeypatch.setenv('PYANNOTE_METRICS_ENABLED', '1')
    monkeypatch.setenv('HF_HUB_OFFLINE', '0')
    enforce_offline()
    assert os.environ['PYANNOTE_METRICS_ENABLED'] == '0'
    assert os.environ['HF_HUB_OFFLINE'] == '1'


def test_missing_model_check_cannot_open_network():
    code = '''import socket
import whisper_daemon
socket.socket.connect = lambda *args: (_ for _ in ()).throw(AssertionError('network attempted'))
from whisper_daemon.model_policy import local_model
try: local_model('whisper-daemon-test/nonexistent-local-model')
except RuntimeError as exc: assert 'not installed locally' in str(exc)
else: raise AssertionError('nonexistent model found')
'''
    assert subprocess.run([sys.executable, '-c', code]).returncode == 0


def test_missing_vad_never_downloads_during_normal_startup(tmp_path, monkeypatch):
    from whisper_daemon import vad
    monkeypatch.setattr(vad, 'MODEL_PATH', tmp_path / 'vad.onnx')
    monkeypatch.setattr(vad, 'MODEL_CACHE_DIR', tmp_path)
    monkeypatch.delenv('WHISPER_PREPARE_MODELS', raising=False)
    calls = []
    monkeypatch.setattr(vad.urllib.request, 'urlretrieve', lambda *args: calls.append(args))
    with pytest.raises(RuntimeError, match='models prepare'):
        vad._ensure_model()
    assert calls == []
