#!/usr/bin/env python3
"""Headless whisper-daemon for end-to-end extension tests.

Runs the real BrowserAudioBridge (thread + event loop + pairing + origin
checks) and the real MenuBarDelegate browser/meeting methods: connect gating,
PCM staging, stop handling, status pushes, the meeting worker, RecordingStore
and exports. Only AppKit UI, audio devices, VAD and Whisper are replaced.

Protocol: JSON lines. Events on stdout, commands on stdin:
  {"cmd": "state"}                       -> {"event": "state", ...}
  {"cmd": "native_stop"}                 stop the meeting from the "menu bar"
  {"cmd": "block", "on": true|false}     drop the socket and refuse new ones
  {"cmd": "set", "transcribe_delay": 1}  seconds each chunk spends in "Whisper"
  {"cmd": "set", "finalize_delay": 2.0}  seconds of silent post-processing (diarization/review)
  {"cmd": "shutdown"}
"""
import argparse
import http
import json
import logging
import os
import queue
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from whisper_daemon import menubar, meeting_recorder, meeting_review, transcriber
from whisper_daemon.audio_server import BrowserAudioBridge
from whisper_daemon.config import Settings
from whisper_daemon.pairing import pair

logger = logging.getLogger("daemon_harness")
EXTENSION_ID = "a" * 32

_BORROWED = (
    "_on_browser_connect", "_on_browser_audio", "_on_browser_disconnect",
    "_on_browser_control", "_handle_get_settings", "_push_settings_snapshot",
    "_run_off_bridge", "_start_meeting", "_stop_meeting", "_meeting_worker",
    "_reset_meeting_ui", "_send_results_to_browser", "_attach_recording_store",
)


class MainThread:
    """Ordered stand-in for the Cocoa main thread behind AppHelper.callAfter."""

    def __init__(self, latency: float = 0.0) -> None:
        self.latency = latency  # a busy Cocoa run loop: menus, redraws, notifications
        self._queue: queue.Queue = queue.Queue()
        threading.Thread(target=self._run, name="fake-main", daemon=True).start()

    def callAfter(self, fn, *args) -> None:
        self._queue.put((fn, args))

    def _run(self) -> None:
        while True:
            fn, args = self._queue.get()
            time.sleep(self.latency)
            try:
                fn(*args)
            except Exception:
                logger.exception("Main-thread callback failed")


class SpeechVAD:
    """Every audible block is speech, so stops produce real transcription jobs."""

    def reset_states(self) -> None:
        pass

    def __call__(self, samples: np.ndarray) -> float:
        return 0.9


class _Inert:
    def __getattr__(self, name: str):
        return lambda *args, **kwargs: None


class HeadlessDaemon:
    """MenuBarDelegate state with the real bridge/meeting methods bound to it."""

    def __init__(self, settings: Settings, bridge_factory) -> None:
        self._settings = settings
        self._daemon = SimpleNamespace(_model=None, _settings=settings)
        self._meeting_active = False
        self._meeting_start = 0.0
        self._meeting_stopped_at = None
        self._meeting_thread = None
        self._meeting_store = None
        self._meeting_paused = False
        self._meeting_pause_events = []
        self._diarization_status = {"available": False, "message": "disabled"}
        self._browser_audio_lock = threading.RLock()
        self._meeting_browser_triggered = False
        self._meeting_phase = "idle"
        self._meeting_chunk_count = 0
        self._results_sent_count = 0
        self._browser_recorder = None
        self._browser_prebuffer = []
        self._browser_prebuffer_bytes = 0
        self._meeting_session_id = None
        self._browser_start_pending = None
        self._last_meeting_dir = None
        self._meeting_menu_item = _Inert()
        self._status_menu_item = _Inert()
        self._browser_bridge = bridge_factory(self)

    def _set_icon(self, *args) -> None:
        pass

    def _set_icon_by_name(self, *args) -> None:
        pass

    def _list_input_devices(self) -> list[str]:
        return []


for _name in _BORROWED:
    setattr(HeadlessDaemon, _name, menubar.MenuBarDelegate.__dict__[_name])


class BlockableBridge(BrowserAudioBridge):
    """Real bridge whose handshake can be refused to simulate an unreachable daemon."""

    blocked = False

    def _process_request(self, connection, request):
        if self.blocked:
            return connection.respond(http.HTTPStatus.SERVICE_UNAVAILABLE, "blocked\n")
        return super()._process_request(connection, request)

    def drop_connection(self) -> None:
        loop, ws = self._loop, self._ws
        if loop is not None and ws is not None:
            loop.call_soon_threadsafe(lambda: loop.create_task(ws.close(1012, "harness drop")))


def _recordings(root: Path) -> list[dict]:
    result = []
    for directory in sorted(root.glob("recording_*"), key=lambda d: d.stat().st_birthtime):
        meta_path = directory / "session.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        frames = {source: (directory / f"{source}.f32").stat().st_size // 4
                  for source in ("mic", "browser") if (directory / f"{source}.f32").exists()}
        result.append({"dir": directory.name, "state": meta.get("state"), "gaps": meta.get("gaps", []),
                       "duration": meta.get("duration"), "frames": frames})
    return result


class Harness:
    """A started headless daemon: the bridge listens and meetings run for real."""

    def __init__(self, port: int, workdir: Path, pairing_path: Path, timing: dict,
                 main_latency: float = 0.0, patch=setattr) -> None:
        self.recordings = workdir / "recordings"
        self.recordings.mkdir(parents=True, exist_ok=True)
        self.pairing = (json.loads(pairing_path.read_text()) if pairing_path.exists()
                        else pair(EXTENSION_ID, pairing_path))
        self.timing = timing
        self.notifications: list[list[str]] = []
        self.main_thread = MainThread(main_latency)
        patch(menubar, "AppHelper", self.main_thread)
        patch(menubar, "_notify", lambda *parts: self.notifications.append([str(p) for p in parts]))
        patch(meeting_recorder, "SileroVAD", SpeechVAD)

        def fake_transcribe(audio, *args, **kwargs):
            time.sleep(timing["transcribe_delay"])
            return {"text": "", "segments": [], "language": "en"}

        real_prepare_review = meeting_review.prepare_review

        def slow_prepare_review(directory):
            time.sleep(timing["finalize_delay"])  # stands in for Whisper + diarization time
            return real_prepare_review(directory)

        patch(transcriber, "transcribe_full", fake_transcribe)
        patch(meeting_review, "prepare_review", slow_prepare_review)
        settings = Settings(recording_dir=str(self.recordings), capture_mic=False, capture_tab=True,
                            diarize=False, capture_screenshots=False, server_port=port)

        def bridge_factory(daemon: HeadlessDaemon) -> BlockableBridge:
            return BlockableBridge("127.0.0.1", port, on_connect=daemon._on_browser_connect,
                                   on_audio=daemon._on_browser_audio, on_disconnect=daemon._on_browser_disconnect,
                                   on_control=daemon._on_browser_control, pairing_path=pairing_path)

        self.daemon = HeadlessDaemon(settings, bridge_factory)
        self.bridge = self.daemon._browser_bridge
        self.bridge.start()
        deadline = time.monotonic() + 10
        while self.bridge._loop is None or not self.bridge._loop.is_running():
            if time.monotonic() > deadline:
                raise RuntimeError("Bridge did not start")
            time.sleep(0.01)
        time.sleep(0.1)

    def state(self) -> dict:
        daemon, bridge = self.daemon, self.bridge
        return {"phase": daemon._meeting_phase, "active": daemon._meeting_active,
                "session_id": bridge.session_id, "connected": bridge.connected,
                "start_pending": daemon._browser_start_pending,
                "closed_sessions": sorted(bridge._closed_sessions),
                "worker_alive": bool(daemon._meeting_thread and daemon._meeting_thread.is_alive()),
                "recordings": _recordings(self.recordings), "notifications": self.notifications}


def emit(event: str, **payload) -> None:
    sys.stdout.write(json.dumps({"event": event, **payload}) + "\n")
    sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--finalize-delay", type=float, default=0.0)
    parser.add_argument("--transcribe-delay", type=float, default=0.0)
    parser.add_argument("--main-latency", type=float, default=0.0)
    parser.add_argument("--pairing", type=Path, default=None, help="reuse this pairing file across restarts")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s [harness] %(name)s: %(message)s")

    workdir = args.workdir or Path(tempfile.mkdtemp(prefix="wd-harness-"))
    workdir.mkdir(parents=True, exist_ok=True)
    timing = {"finalize_delay": args.finalize_delay, "transcribe_delay": args.transcribe_delay}
    harness = Harness(args.port, workdir, args.pairing or workdir / "pairing.json", timing, args.main_latency)
    bridge = harness.bridge
    emit("ready", port=args.port, secret=harness.pairing["secret"], extension_id=EXTENSION_ID, workdir=str(workdir))

    for line in sys.stdin:
        command = json.loads(line)
        kind = command.get("cmd")
        if kind == "state":
            emit("state", **harness.state())
        elif kind == "native_stop":
            harness.main_thread.callAfter(harness.daemon._stop_meeting)
            emit("ok", cmd=kind)
        elif kind == "block":
            bridge.blocked = bool(command.get("on"))
            if bridge.blocked:
                bridge.drop_connection()
            emit("ok", cmd=kind)
        elif kind == "set":
            timing.update({k: float(v) for k, v in command.items() if k in timing})
            emit("ok", cmd=kind)
        elif kind == "shutdown":
            bridge.stop()
            emit("ok", cmd=kind)
            os._exit(0)  # the meeting worker is a non-daemon thread by design


if __name__ == "__main__":
    main()
