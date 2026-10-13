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

from .bridge_protocol import is_allowed_origin, parse_client_message, validate_client_message, AudioSequence
from .pairing import load_pairing
import secrets

logger = logging.getLogger(__name__)

# Control message types routed to the on_control callback instead of the
# start/stop/ping audio-session handlers.
_CONTROL_TYPES = frozenset({"get_settings", "set_settings", "stop_meeting", "pause_meeting"})


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
        pairing_path=None,
    ) -> None:
        if host not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("The audio bridge must bind to loopback")
        self._pairing_path = pairing_path
        self._session_id = None
        self._audio_sequence = AudioSequence()
        self._closed_sessions = set()
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

    def send_json_str(self, msg: str) -> None:
        """Send a pre-built JSON message (settings/status) — thread-safe."""
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
            max_size=65552,
            max_queue=16,
            compression=None,
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
        if is_allowed_origin(origin, load_pairing(self._pairing_path).get("extension_id")):
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

        if is_allowed_origin(origin, load_pairing(self._pairing_path).get("extension_id")):
            return True

        logger.warning("Rejecting connection from disallowed origin: %r", origin)
        try:
            await ws.close(1008, "forbidden origin")
        except Exception:
            logger.debug("Error closing rejected connection", exc_info=True)
        return False

    def end_session(self):
        """Called when native controls finish a meeting, preventing reconnect restart."""
        if self._session_id:
            self._closed_sessions.add(self._session_id)
            if len(self._closed_sessions) > 100: self._closed_sessions.pop()
        self._session_id = None

    async def _handler(self, ws) -> None:
        if not await self._origin_allowed(ws): return
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=5)
            auth = parse_client_message(raw) if isinstance(raw, str) else None
            pairing = load_pairing(self._pairing_path)
            if (not auth or auth.get("type") != "auth" or not isinstance(auth.get("secret"), str)
                    or not pairing.get("secret")
                    or not secrets.compare_digest(auth["secret"], pairing["secret"])):
                await ws.close(1008, "Pair this extension using whisper-daemon pair")
                return
        except (TimeoutError, websockets.exceptions.ConnectionClosed):
            await ws.close(1008, "Authentication required")
            return
        old = self._ws
        self._ws = ws
        if old is not None: await old.close()
        await ws.send(json.dumps({"type": "authenticated"}))
        attached_session = None
        try:
            async for message in ws:
                if isinstance(message, bytes):
                    if not attached_session or attached_session != self._session_id:
                        raise ValueError("Audio outside an active session")
                    accepted = self._audio_sequence.accept(message)
                    if accepted:
                        audio, gap = accepted
                        if gap:
                            if self._on_control:
                                self._on_control({"type": "audio_gap", "samples": gap,
                                                  "end": (self._audio_sequence.frames - len(audio) // 4) / 16000})
                            while gap:
                                size = min(gap, 4096)
                                self._on_audio(bytes(size * 4))
                                gap -= size
                        self._on_audio(audio)
                    await ws.send(json.dumps({"type": "audio_ack", "session_id": self._session_id,
                                               "sequence": self._audio_sequence.sequence}))
                    continue
                if len(message) > 16384: raise ValueError("Control frame too large")
                data = validate_client_message(parse_client_message(message))
                kind = data['type']
                if kind == 'start':
                    session = data['session_id']
                    if session in self._closed_sessions:
                        raise ValueError("This meeting has ended; start a new capture")
                    if self._session_id and self._session_id != session:
                        raise ValueError("Another meeting owns the audio bridge")
                    if self._session_id is None:
                        accepted = self._on_connect(data.get('meeting_title', ''), data.get('meeting_url', ''))
                        if accepted is False: raise ValueError("Meeting is still being finalized")
                        self._session_id = session
                        self._audio_sequence = AudioSequence()
                    attached_session = session
                    await ws.send(json.dumps({'type': 'started', 'session_id': session}))
                elif kind == 'stop':
                    if data['session_id'] == self._session_id:
                        self.end_session()
                        self._on_disconnect()
                elif kind == 'ping':
                    await ws.send(json.dumps({'type': 'pong'}))
                elif kind in _CONTROL_TYPES and self._on_control:
                    self._on_control(data)
        except ValueError as exc:
            await ws.send(json.dumps({'type': 'error', 'message': str(exc)}))
            await ws.close(1008, str(exc)[:100])
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception:
            logger.exception('Browser audio failed')
        finally:
            if self._ws is ws:
                self._ws = None
                # Keep the recording and sample sequence alive on transport
                # loss. A reconnect resumes the same session; native Stop
                # remains available. Never start a second meeting implicitly.
                logger.info('Browser disconnected; microphone recording continues')
