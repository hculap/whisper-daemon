"""FastAPI WebSocket server for receiving audio from browser extensions."""

import asyncio
import concurrent.futures
import json
import logging
import queue
from datetime import datetime
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from whisper_daemon.formats import FORMATTERS
from whisper_daemon.meeting_recorder import AudioChunk, MeetingRecorder
from whisper_daemon.transcriber import transcribe as transcribe_audio

logger = logging.getLogger(__name__)

app = FastAPI()


class RecordingSession:
    """Manages a single browser recording session."""

    def __init__(
        self,
        model: str,
        output_dir: Path,
        formats: list[str],
        language: str | None,
        chunk_silence: float,
        diarize: bool,
        diarize_mode: str,
        num_speakers: int | None,
    ) -> None:
        self._model = model
        self._output_dir = output_dir
        self._formats = formats
        self._language = language
        self._diarize = diarize
        self._diarize_mode = diarize_mode
        self._num_speakers = num_speakers

        self._chunk_queue: queue.Queue[AudioChunk | None] = queue.Queue()
        self._recorder = MeetingRecorder(
            self._chunk_queue, chunk_silence=chunk_silence,
        )

        self._all_results: list[tuple[float, dict]] = []
        self._all_audio_chunks: list[tuple[float, np.ndarray]] = []
        self._chunks_transcribed = 0
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._futures: dict[concurrent.futures.Future, float] = {}
        self._speaker_tracker = None
        self._meeting_title = ""
        self._meeting_url = ""

    def start(self, meeting_title: str = "", meeting_url: str = "") -> None:
        self._meeting_title = meeting_title
        self._meeting_url = meeting_url

        if self._diarize and self._diarize_mode in ("realtime", "hybrid"):
            from whisper_daemon.diarizer import SpeakerTracker
            self._speaker_tracker = SpeakerTracker()

        self._recorder.start_without_device()
        logger.info(
            "Recording session started — title=%s, url=%s",
            meeting_title, meeting_url,
        )

    def feed_audio(self, data: bytes) -> None:
        samples = np.frombuffer(data, dtype=np.float32)
        self._recorder.feed_audio(samples)

    def stop(self) -> None:
        self._recorder.stop_without_device()
        logger.info("Recording session stopped")

    def process_pending_chunks(self) -> list[dict]:
        """Process any chunks ready for transcription. Returns new results."""
        new_results = []

        while True:
            try:
                chunk = self._chunk_queue.get_nowait()
            except queue.Empty:
                break

            if chunk is None:
                break

            logger.info(
                "Chunk %d: %.1fs at %.1fs",
                self._chunks_transcribed + 1, chunk.duration, chunk.start_time,
            )

            if self._diarize:
                self._all_audio_chunks.append((chunk.start_time, chunk.audio.copy()))

            if self._speaker_tracker is not None:
                self._speaker_tracker.identify(chunk.audio, chunk.start_time)

            future = self._pool.submit(transcribe_audio, chunk.audio, self._model)
            self._futures[future] = chunk.start_time

        done = [f for f in self._futures if f.done()]
        for future in done:
            start_time = self._futures.pop(future)
            try:
                result = future.result()
                if result:
                    self._all_results.append((start_time, {"text": result, "segments": [], "language": ""}))
                    self._chunks_transcribed += 1
                    new_results.append({
                        "text": result,
                        "start_time": start_time,
                        "chunk_index": self._chunks_transcribed,
                    })
            except Exception:
                logger.exception("Transcription failed for chunk at %.1fs", start_time)

        return new_results

    def finalize(self) -> Path | None:
        """Wait for remaining transcriptions and write output files."""
        for future in concurrent.futures.as_completed(self._futures):
            start_time = self._futures[future]
            try:
                result = future.result()
                if result:
                    self._all_results.append((start_time, {"text": result, "segments": [], "language": ""}))
                    self._chunks_transcribed += 1
            except Exception:
                logger.exception("Transcription failed for chunk at %.1fs", start_time)

        self._futures.clear()
        self._pool.shutdown(wait=False)

        if not self._all_results:
            logger.info("No audio transcribed")
            return None

        self._all_results.sort(key=lambda r: r[0])

        if self._diarize and self._diarize_mode in ("batch", "hybrid") and self._all_audio_chunks:
            self._run_batch_diarization()

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        title_slug = self._meeting_title.replace(" ", "_")[:30] if self._meeting_title else "meeting"
        stem = f"{title_slug}_{timestamp}"

        merged = {
            "text": "\n\n".join(r["text"] for _, r in self._all_results),
            "segments": [],
            "language": "",
        }

        output_path = None
        for fmt in self._formats:
            formatter = FORMATTERS.get(fmt)
            if not formatter:
                continue
            text = formatter(merged)
            path = self._output_dir / f"{stem}.{fmt}"
            path.write_text(text, encoding="utf-8")
            logger.info("Transcript saved: %s", path)
            if output_path is None:
                output_path = path

        return output_path

    def _run_batch_diarization(self) -> None:
        try:
            from whisper_daemon.diarizer import diarize_batch
            all_audio = np.concatenate([a for _, a in self._all_audio_chunks])
            segments = diarize_batch(all_audio, num_speakers=self._num_speakers)
            if segments:
                logger.info("Batch diarization: %d segments, %d speakers",
                            len(segments), len({s.speaker for s in segments}))
        except Exception:
            logger.exception("Batch diarization failed")

    @property
    def chunks_transcribed(self) -> int:
        return self._chunks_transcribed


_active_session: RecordingSession | None = None
_server_config: dict = {}


def configure_server(
    model: str,
    output_dir: str,
    formats: list[str],
    language: str | None,
    chunk_silence: float,
    diarize: bool,
    diarize_mode: str,
    num_speakers: int | None,
) -> None:
    """Store server config for use by WebSocket handler."""
    global _server_config
    _server_config = {
        "model": model,
        "output_dir": Path(output_dir).expanduser(),
        "formats": formats,
        "language": language,
        "chunk_silence": chunk_silence,
        "diarize": diarize,
        "diarize_mode": diarize_mode,
        "num_speakers": num_speakers,
    }
    _server_config["output_dir"].mkdir(parents=True, exist_ok=True)


@app.websocket("/ws/audio")
async def audio_websocket(ws: WebSocket) -> None:
    global _active_session

    await ws.accept()
    logger.info("WebSocket client connected")

    if _active_session is not None:
        await ws.send_json({"type": "error", "message": "Recording already in progress"})
        await ws.close()
        return

    session: RecordingSession | None = None

    try:
        while True:
            message = await ws.receive()

            if message["type"] == "websocket.disconnect":
                break

            if "bytes" in message and message["bytes"]:
                if session is not None:
                    session.feed_audio(message["bytes"])

                    new_results = session.process_pending_chunks()
                    for result in new_results:
                        await ws.send_json({
                            "type": "chunk_transcribed",
                            "text": result["text"],
                            "start_time": result["start_time"],
                            "chunk_index": result["chunk_index"],
                        })
                continue

            if "text" in message and message["text"]:
                try:
                    data = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue

                msg_type = data.get("type")

                if msg_type == "start":
                    if session is not None:
                        await ws.send_json({"type": "error", "message": "Already recording"})
                        continue

                    session = RecordingSession(**_server_config)
                    _active_session = session
                    session.start(
                        meeting_title=data.get("meeting_title", ""),
                        meeting_url=data.get("meeting_url", ""),
                    )
                    await ws.send_json({"type": "status", "recording": True, "chunks_transcribed": 0})

                elif msg_type == "stop":
                    if session is not None:
                        session.stop()

                        loop = asyncio.get_event_loop()
                        output_path = await loop.run_in_executor(None, session.finalize)

                        await ws.send_json({
                            "type": "status",
                            "recording": False,
                            "chunks_transcribed": session.chunks_transcribed,
                            "output_path": str(output_path) if output_path else None,
                        })
                        session = None
                        _active_session = None

                elif msg_type == "ping":
                    await ws.send_json({"type": "pong"})

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception:
        logger.exception("WebSocket error")
    finally:
        if session is not None:
            session.stop()
            session.finalize()
            _active_session = None
        logger.info("WebSocket connection closed")
