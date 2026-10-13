"""Offline inference by default; model provisioning is an explicit CLI action."""
import os
from pathlib import Path


def enforce_offline():
    # Must run before importing MLX, HF Hub, transformers or pyannote.
    os.environ['PYANNOTE_METRICS_ENABLED'] = '0'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['DO_NOT_TRACK'] = '1'
    if os.environ.get('WHISPER_PREPARE_MODELS') != '1':
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'


def local_model(model):
    if Path(model).is_dir(): return model
    from huggingface_hub import snapshot_download
    try:
        return snapshot_download(model, local_files_only=True)
    except Exception as exc:
        raise RuntimeError(f'Model is not installed locally: {model}. Run whisper-daemon models prepare.') from exc


def diarization_status():
    try:
        local_model("pyannote/speaker-diarization-community-1")
        from whisper_daemon.diarizer import _get_pipeline
        _get_pipeline()  # includes dependent models and actual load compatibility
        return {'available': True, 'message': 'Speaker recognition ready (offline)'}
    except Exception:
        return {'available': False, 'message': 'Speaker recognition unavailable: install authorized models with whisper-daemon models prepare --diarization. Audio and transcript can still be recorded.'}
