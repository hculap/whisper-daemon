"""Lightweight WebSocket server that relays browser audio to the daemon.

The BrowserAudioBridge runs a websockets server in a background thread.
It accepts one connection at a time from the Chrome extension and
forwards events and PCM audio to the daemon via callbacks.
"""

import asyncio
import json
import logging
import threading
from collections.abc import Callable

import websockets
import websockets.asyncio.server

logger = logging.getLogger(__name__)


class BrowserAudioBridge:
    """WebSocket server relaying browser audio + meeting events to the daemon."""

    def __init__(
        self,
        host: str,
        port: int,
        on_connect: Callable[[str, str], None],
        on_audio: Callable[[bytes], None],
        on_disconnect: Callable[[], None],
    ) -> None:
        self._host = host
        self._port = port
        self._on_connect = on_connect
        self._on_audio = on_audio
        self._on_disconnect = on_disconnect

        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_event: asyncio.Event | None = None
        self._ws: websockets.asyncio.server.ServerConnection | None = None

    def start(self) -> None:
        """Start the WebSocket server in a daemon thread."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("Browser bridge starting on ws://%s:%d", self._host, self._port)

    def stop(self) -> None:
        """Shut down the WebSocket server."""
        if self._loop is not None and self._stop_event is not None:
            self._loop.call_soon_threadsafe(self._stop_event.set)
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None
        logger.info("Browser bridge stopped")

    @property
    def connected(self) -> bool:
        return self._ws is not None

    def send_chunk_result(self, text: str, start_time: float, chunk_index: int) -> None:
        """Send a transcription result back to the extension (thread-safe)."""
        if self._ws is None or self._loop is None:
            return
        msg = json.dumps({
            "type": "chunk_transcribed",
            "text": text,
            "start_time": start_time,
            "chunk_index": chunk_index,
        })
        self._loop.call_soon_threadsafe(
            asyncio.ensure_future,
            self._safe_send(msg),
        )

    def send_status(self, recording: bool, chunks_transcribed: int) -> None:
        """Send a status update to the extension (thread-safe)."""
        if self._ws is None or self._loop is None:
            return
        msg = json.dumps({
            "type": "status",
            "recording": recording,
            "chunks_transcribed": chunks_transcribed,
        })
        self._loop.call_soon_threadsafe(
            asyncio.ensure_future,
            self._safe_send(msg),
        )

    async def _safe_send(self, msg: str) -> None:
        try:
            if self._ws is not None:
                await self._ws.send(msg)
        except Exception:
            logger.debug("Failed to send to extension", exc_info=True)

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._stop_event = asyncio.Event()
        try:
            self._loop.run_until_complete(self._serve())
        except Exception:
            logger.exception("Browser bridge server error")
        finally:
            self._loop.close()
            self._loop = None

    async def _serve(self) -> None:
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            ping_interval=10,
            ping_timeout=15,
            close_timeout=5,
        ):
            logger.info("Browser bridge listening on ws://%s:%d", self._host, self._port)
            await self._stop_event.wait()

    async def _handler(self, ws: websockets.asyncio.server.ServerConnection) -> None:
        if self._ws is not None:
            await ws.send(json.dumps({
                "type": "error",
                "message": "Another browser session is already connected",
            }))
            await ws.close()
            return

        self._ws = ws
        logger.info("Browser extension connected")

        try:
            async for message in ws:
                if isinstance(message, bytes):
                    self._on_audio(message)
                    continue

                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    continue

                msg_type = data.get("type")

                if msg_type == "start":
                    self._on_connect(
                        data.get("meeting_title", ""),
                        data.get("meeting_url", ""),
                    )

                elif msg_type == "stop":
                    self._on_disconnect()

                elif msg_type == "ping":
                    await ws.send(json.dumps({"type": "pong"}))

        except websockets.exceptions.ConnectionClosed:
            logger.info("Browser extension disconnected")
        except Exception:
            logger.exception("Browser bridge handler error")
        finally:
            self._ws = None
            self._on_disconnect()
            logger.info("Browser extension connection closed")
