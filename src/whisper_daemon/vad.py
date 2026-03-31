"""Silero VAD using ONNX Runtime directly — no torch dependency."""

import logging
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

MODEL_URL = "https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx"
MODEL_CACHE_DIR = Path.home() / ".cache" / "whisper-daemon"
MODEL_PATH = MODEL_CACHE_DIR / "silero_vad.onnx"

SAMPLE_RATE = 16000
CHUNK_SIZE = 512
CONTEXT_SIZE = 64
INPUT_SIZE = CHUNK_SIZE + CONTEXT_SIZE  # 576
HIDDEN_DIM = 128


class SileroVAD:
    """Lightweight Silero VAD wrapper using only numpy + onnxruntime."""

    def __init__(self) -> None:
        model_path = _ensure_model()
        self._session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)
        self.reset_states()
        logger.info("Silero VAD loaded (ONNX-only, no torch)")

    def reset_states(self) -> None:
        """Reset hidden state and context buffer for a new audio stream."""
        self._state = np.zeros((2, 1, HIDDEN_DIM), dtype=np.float32)
        self._context = np.zeros((1, CONTEXT_SIZE), dtype=np.float32)

    def __call__(self, chunk: np.ndarray) -> float:
        """Run VAD on a 512-sample audio chunk.

        Args:
            chunk: 1D float32 array of exactly 512 samples, normalized to [-1, 1].

        Returns:
            Speech probability (0.0 to 1.0).
        """
        chunk_2d = chunk.reshape(1, -1)
        input_tensor = np.concatenate([self._context, chunk_2d], axis=1)

        ort_inputs = {
            "input": input_tensor,
            "state": self._state,
            "sr": self._sr,
        }
        output, new_state = self._session.run(None, ort_inputs)

        self._state = new_state
        self._context = input_tensor[:, -CONTEXT_SIZE:]

        return float(output[0][0])


def _ensure_model() -> Path:
    """Download the Silero VAD ONNX model if not cached."""
    if MODEL_PATH.exists():
        return MODEL_PATH

    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading Silero VAD ONNX model (~2MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    logger.info("Model saved to %s", MODEL_PATH)
    return MODEL_PATH
