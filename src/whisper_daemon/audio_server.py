"""Lightweight WebSocket server that relays browser audio to the daemon.

The BrowserAudioBridge runs a websockets server in a background thread.
It accepts one connection at a time from the Chrome extension and
forwards events and PCM audio to the daemon via callbacks.
"""

import asyncio
import http
import json
import logging
import threading
import time
from collections.abc import Callable

import websockets
import websockets.asyncio.server

from .bridge_protocol import is_allowed_origin

logger = logging.getLogger(__name__)

# Control message types routed to the on_control callback instead of the
# start/stop/ping audio-session handlers.
_CONTROL_TYPES = frozenset({"get_settings", "set_settings", "stop_meeting"})


class BrowserAudioBridge:
    """WebSocket server relaying browser audio + meeting events to the daemon."""

    def __init__(
        self,
        host: str,
        port: int,
        on_connect: Callable[[str, str], None],
        on_audio: Callable[[bytes], None],
        on_disconnect: Callable[[], None],
        on_control: Callable[[dict], None] | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._on_connect = on_connect
        self._on_audio = on_audio
        self._on_disconnect = on_disconnect
        self._on_control = on_control

        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_event: asyncio.Event | None = None
        self._ws: websockets.asyncio.server.ServerConnection | None = None
        self._stopping = False

    def start(self) -> None:
        """Start the WebSocket server in a daemon thread."""
        self._stopping = False
        self._thread = threading.Thread(
            target=self._run, name="browser-bridge", daemon=True
        )
        self._thread.start()
        logger.info("Browser bridge starting on ws://%s:%d", self._host, self._port)

    def stop(self) -> None:
        """Shut down the WebSocket server."""
        self._stopping = True
        loop, stop_event = self._loop, self._stop_event
        if loop is not None and stop_event is not None:
            try:
                loop.call_soon_threadsafe(stop_event.set)
            except RuntimeError:
                pass  # loop already closed
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None
        logger.info("Browser bridge stopped")

    @property
    def connected(self) -> bool:
        return self._ws is not None

    def _schedule_send(self, msg: str) -> None:
        """Schedule a send on the bridge loop without ever raising.

        Called from the meeting worker thread. The serve-retry loop can
        null or close ``self._loop`` between our check and the scheduling
        call, so capture it locally and swallow the RuntimeError that a
        closed loop raises — a dropped status message must never abort the
        caller's meeting.
        """
        loop = self._loop
        if self._ws is None or loop is None:
            return
        try:
            loop.call_soon_threadsafe(asyncio.ensure_future, self._safe_send(msg))
        except RuntimeError:
            logger.debug("Bridge loop unavailable — dropping message", exc_info=True)

    def send_chunk_result(self, text: str, start_time: float, chunk_index: int) -> None:
        """Send a transcription result back to the extension (thread-safe)."""
        self._schedule_send(json.dumps({
            "type": "chunk_transcribed",
            "text": text,
            "start_time": start_time,
            "chunk_index": chunk_index,
        }))

    def send_status(self, recording: bool, chunks_transcribed: int) -> None:
        """Send a status update to the extension (thread-safe)."""
        self._schedule_send(json.dumps({
            "type": "status",
            "recording": recording,
            "chunks_transcribed": chunks_transcribed,
        }))

    def send_settings(self, msg: str) -> None:
        """Send a pre-built 'settings' JSON message (thread-safe)."""
        self._schedule_send(msg)

    def send_status_msg(self, msg: str) -> None:
        """Send a pre-built 'status' JSON message (thread-safe)."""
        self._schedule_send(msg)

    async def _safe_send(self, msg: str) -> None:
        try:
            if self._ws is not None:
                await self._ws.send(msg)
        except Exception:
            logger.debug("Failed to send to extension", exc_info=True)

    def _run(self) -> None:
        """Serve forever, restarting after errors.

        A bind failure (port still held after an unclean restart) or an
        unexpected event-loop death must not kill the extension surface
        for the daemon's whole lifetime — retry with backoff until
        stop() is called.
        """
        backoff = 1.0
        while not self._stopping:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._stop_event = asyncio.Event()
            try:
                self._loop.run_until_complete(self._serve())
                break  # clean stop() shutdown
            except Exception:
                logger.exception(
                    "Browser bridge server error — retrying in %.0fs", backoff
                )
            finally:
                self._ws = None
                self._loop.close()
                self._loop = None
            if self._stopping:
                break
            time.sleep(backoff)
            backoff = min(backoff * 2, 30.0)

    async def _serve(self) -> None:
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            process_request=self._process_request,
            ping_interval=10,
            ping_timeout=15,
            close_timeout=5,
        ):
            logger.info("Browser bridge listening on ws://%s:%d", self._host, self._port)
            await self._stop_event.wait()

    def _process_request(self, connection, request):
        """Reject the WebSocket handshake itself for disallowed origins.

        Runs during the HTTP upgrade, before the 101 is sent — so a website
        never gets an open socket at all (returning a 403 aborts the upgrade).
        The in-handler _origin_allowed check remains as defense in depth.
        """
        origin = request.headers.get("Origin")
        if is_allowed_origin(origin):
            return None
        logger.warning("Rejecting handshake from disallowed origin: %r", origin)
        return connection.respond(http.HTTPStatus.FORBIDDEN, "forbidden origin\n")

    async def _origin_allowed(
        self, ws: websockets.asyncio.server.ServerConnection
    ) -> bool:
        """Close and reject the socket unless its Origin is a chrome extension."""
        origin = None
        try:
            request = getattr(ws, "request", None)
            if request is not None:
                origin = request.headers.get("Origin")
        except Exception:
            logger.debug("Could not read Origin header", exc_info=True)
            origin = None

        if is_allowed_origin(origin):
            return True

        logger.warning("Rejecting connection from disallowed origin: %r", origin)
        try:
            await ws.close(1008, "forbidden origin")
        except Exception:
            logger.debug("Error closing rejected connection", exc_info=True)
        return False

    async def _handler(self, ws: websockets.asyncio.server.ServerConnection) -> None:
        # Reject any non-extension origin (websites, missing header) BEFORE
        # registering this socket as the active connection, so a malicious web
        # page cannot drive the daemon. The real extension's Origin is
        # chrome-extension://<id> and passes.
        if not await self._origin_allowed(ws):
            return

        old = self._ws
        if old is not None:
            # Take over: the previous connection is likely a zombie (MV3
            # service worker died, network blip) — a live extension only
            # opens a new socket when the old one is gone. Rejecting the
            # newcomer would lock the extension out until ping timeout.
            logger.warning("New browser connection — closing previous one")
            self._ws = None
            try:
                await old.close()
            except Exception:
                logger.debug("Error closing previous connection", exc_info=True)

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

                elif msg_type in _CONTROL_TYPES:
                    if self._on_control is not None:
                        try:
                            self._on_control(data)
                        except Exception:
                            logger.exception("on_control handler failed")

        except websockets.exceptions.ConnectionClosed:
            logger.info("Browser extension disconnected")
        except Exception:
            logger.exception("Browser bridge handler error")
        finally:
            # If another connection already took over, this socket's death
            # is not a session end — don't clobber the new connection or
            # stop the meeting it continues.
            if self._ws is ws:
                self._ws = None
                self._on_disconnect()
                logger.info("Browser extension connection closed")
            else:
                logger.info("Superseded browser connection closed")
