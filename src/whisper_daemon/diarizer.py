"""Speaker diarization using pyannote-audio.

Three approaches — batch, realtime (embedding bank), and hybrid —
so the user can benchmark quality vs speed and choose.
"""

import logging
from dataclasses import dataclass

import numpy as np
import torch

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
PIPELINE_MODEL = "pyannote/speaker-diarization-community-1"
EMBEDDING_MODEL = "pyannote/wespeaker-vox-ceb-resnet34-LM"
SIMILARITY_THRESHOLD = 0.60
MIN_SPEECH_SEC = 2.0
MAX_EMBEDDINGS_PER_SPEAKER = 20

_pipeline = None


def _get_hf_token() -> str | None:
    """Get HuggingFace token from cached login."""
    try:
        from huggingface_hub import get_token
        return get_token()
    except Exception:
        return None


@dataclass(frozen=True)
class SpeakerSegment:
    """A time region attributed to a speaker."""

    speaker: int
    start: float
    end: float


# ---------------------------------------------------------------------------
# Batch diarization (full pyannote pipeline)
# ---------------------------------------------------------------------------


def _get_pipeline():
    """Lazy-load and cache the pyannote pipeline."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    import os

    from pyannote.audio import Pipeline

    token = os.environ.get("HF_TOKEN") or _get_hf_token()

    logger.info("Loading pyannote pipeline (%s)...", PIPELINE_MODEL)
    _pipeline = Pipeline.from_pretrained(PIPELINE_MODEL, token=token)

    if torch.backends.mps.is_available():
        _pipeline.to(torch.device("mps"))
        logger.info("Pipeline moved to MPS (Apple GPU)")
    else:
        logger.info("Pipeline running on CPU")

    return _pipeline


def _result_to_segments(result: object) -> list[SpeakerSegment]:
    """Convert pipeline output to SpeakerSegment list.

    Handles both the new DiarizeOutput (community-1) which has
    .exclusive_speaker_diarization and legacy Annotation objects.
    """
    # community-1 returns DiarizeOutput with exclusive (non-overlapping) annotation
    if hasattr(result, "exclusive_speaker_diarization"):
        annotation = result.exclusive_speaker_diarization
    elif hasattr(result, "speaker_diarization"):
        annotation = result.speaker_diarization
    else:
        annotation = result  # legacy Annotation

    speaker_map: dict[str, int] = {}
    segments: list[SpeakerSegment] = []

    for segment, _track, speaker_label in annotation.itertracks(yield_label=True):
        if speaker_label not in speaker_map:
            speaker_map[speaker_label] = len(speaker_map)

        segments.append(SpeakerSegment(
            speaker=speaker_map[speaker_label],
            start=segment.start,
            end=segment.end,
        ))

    segments.sort(key=lambda s: s.start)
    logger.info(
        "Diarization done — %d segments, %d speakers",
        len(segments),
        len(speaker_map),
    )
    return segments


def diarize_batch(
    audio: np.ndarray,
    num_speakers: int | None = None,
) -> list[SpeakerSegment]:
    """Run full pyannote diarization on complete audio.

    Args:
        audio: 1D float32, 16 kHz mono.
        num_speakers: Optional speaker count hint.

    Returns:
        Speaker segments sorted by start time.
    """
    pipeline = _get_pipeline()

    waveform = torch.from_numpy(audio).unsqueeze(0).float()
    input_data = {"waveform": waveform, "sample_rate": SAMPLE_RATE}

    kwargs = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers

    logger.info(
        "Diarizing %.1fs of audio (batch mode)...",
        len(audio) / SAMPLE_RATE,
    )
    try:
        result = pipeline(input_data, **kwargs)
    except ValueError as exc:
        if "samples instead of the expected" in str(exc):
            logger.warning("Audio too short for diarization: %s", exc)
            return []
        raise
    return _result_to_segments(result)


def diarize_file(
    path: str,
    num_speakers: int | None = None,
) -> list[SpeakerSegment]:
    """Run diarization on an audio/video file (ffmpeg decodes it)."""
    pipeline = _get_pipeline()

    kwargs = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers

    logger.info("Diarizing file: %s", path)
    try:
        result = pipeline(path, **kwargs)
    except ValueError as exc:
        if "samples instead of the expected" in str(exc):
            logger.warning("Audio too short for diarization: %s", exc)
            return []
        raise
    return _result_to_segments(result)


# ---------------------------------------------------------------------------
# Realtime diarization (embedding bank)
# ---------------------------------------------------------------------------

_embedding_model = None


def _get_embedding_model():
    """Get the embedding model from the diarization pipeline.

    Returns the pipeline's bundled PyannoteAudioPretrainedSpeakerEmbedding,
    which is callable: (waveforms, masks) -> embeddings.
    """
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model

    pipeline = _get_pipeline()
    _embedding_model = pipeline._embedding
    logger.info("Speaker embedding model ready (dim=%d)", _embedding_model.dimension)
    return _embedding_model


class SpeakerTracker:
    """Tracks speakers across audio chunks using an embedding bank.

    For each chunk, extracts speaker embeddings and matches against
    known speakers via cosine similarity. New speakers are registered
    when no match exceeds the threshold.
    """

    def __init__(
        self,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
    ) -> None:
        self._threshold = similarity_threshold
        self._embedding_bank: dict[int, list[np.ndarray]] = {}
        self._next_speaker_id = 0
        self._all_segments: list[SpeakerSegment] = []

    def identify(
        self,
        audio: np.ndarray,
        start_time: float,
    ) -> list[SpeakerSegment]:
        """Identify speakers in a chunk and return segments.

        Args:
            audio: 1D float32, 16 kHz mono.
            start_time: Chunk offset in the recording timeline.

        Returns:
            Speaker segments for this chunk.
        """
        duration = len(audio) / SAMPLE_RATE
        if duration < MIN_SPEECH_SEC:
            logger.debug(
                "Chunk too short for embedding (%.1fs < %.1fs), skipping",
                duration,
                MIN_SPEECH_SEC,
            )
            return []

        model = _get_embedding_model()
        # Shape: (batch=1, channel=1, samples)
        waveform = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0).float()

        try:
            embedding = model(waveform)
        except Exception:
            logger.exception("Embedding extraction failed")
            return []

        if isinstance(embedding, torch.Tensor):
            embedding = embedding.cpu().numpy()
        embedding = embedding.flatten()

        speaker_id = self._match_or_register(embedding)

        segment = SpeakerSegment(
            speaker=speaker_id,
            start=start_time,
            end=start_time + duration,
        )
        self._all_segments.append(segment)
        return [segment]

    def get_all_segments(self) -> list[SpeakerSegment]:
        """Return all segments identified so far."""
        return list(self._all_segments)

    @property
    def speaker_count(self) -> int:
        return len(self._embedding_bank)

    def _match_or_register(self, embedding: np.ndarray) -> int:
        """Match embedding against bank or register new speaker."""
        best_speaker = -1
        best_score = -1.0

        for speaker_id, embeddings in self._embedding_bank.items():
            centroid = np.mean(embeddings, axis=0)
            score = _cosine_similarity(embedding, centroid)
            if score > best_score:
                best_score = score
                best_speaker = speaker_id

        if best_score >= self._threshold:
            bank = self._embedding_bank[best_speaker]
            bank.append(embedding)
            if len(bank) > MAX_EMBEDDINGS_PER_SPEAKER:
                self._embedding_bank[best_speaker] = bank[-MAX_EMBEDDINGS_PER_SPEAKER:]
            logger.debug(
                "Speaker %d matched (score=%.3f)",
                best_speaker,
                best_score,
            )
            return best_speaker

        new_id = self._next_speaker_id
        self._next_speaker_id += 1
        self._embedding_bank[new_id] = [embedding]
        logger.info("New speaker registered: Speaker %d", new_id + 1)
        return new_id


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
