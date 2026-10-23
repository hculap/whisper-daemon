"""Daemon side end to end: a real WebSocket client against the real bridge and
the real menubar meeting pipeline (tests/daemon_harness.py)."""
import asyncio
import json
import socket
import struct
import time

import numpy as np
import pytest
from websockets.asyncio.client import connect

from daemon_harness import EXTENSION_ID, Harness


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def packet(sequence: int, position: int, samples: int) -> bytes:
    audio = (np.arange(position, position + samples) % 200 / 1000).astype("<f4")
    return struct.pack("<4sIQ", b"WDA1", sequence, position) + audio.tobytes()


@pytest.fixture
def start_harness(tmp_path, monkeypatch):
    started = []

    def start(main_latency: float = 0.0, **timing) -> Harness:
        harness = Harness(free_port(), tmp_path, tmp_path / "pairing.json",
                          {"finalize_delay": 0.0, "transcribe_delay": 0.0, **timing},
                          main_latency=main_latency, patch=monkeypatch.setattr)
        started.append(harness)
        return harness

    yield start
    for harness in started:
        harness.bridge.stop()
        thread = harness.daemon._meeting_thread
        if thread is not None:
            thread.join(timeout=30)


def settled(harness: Harness, timeout: float = 15) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = harness.state()
        if (state["phase"] == "idle" and not state["worker_alive"] and not state["start_pending"]
                and not state["session_id"] and all(r["state"] == "complete" for r in state["recordings"])):
            return state
        time.sleep(0.05)
    raise AssertionError(f"daemon never settled: {harness.state()}")


async def client(harness: Harness):
    port = harness.bridge._port
    ws = await connect(f"ws://127.0.0.1:{port}", origin=f"chrome-extension://{EXTENSION_ID}")
    await ws.send(json.dumps({"type": "auth", "secret": harness.pairing["secret"]}))
    assert json.loads(await ws.recv())["type"] == "authenticated"
    return ws


async def expect(ws, kind: str, **fields) -> dict:
    """Next message of this type and fields; statuses/health in between are skipped."""
    while True:
        message = json.loads(await asyncio.wait_for(ws.recv(), 10))
        if message["type"] == kind and all(message.get(k) == v for k, v in fields.items()):
            return message


def test_stop_before_the_main_thread_starts_the_meeting_saves_its_audio(start_harness):
    # A recovered tail arrives as start -> PCM -> stop. With a busy main thread
    # the stop reaches the bridge before the meeting starts; the audio must be
    # saved, not dropped.
    harness = start_harness(main_latency=0.2)

    async def deliver_tail():
        ws = await client(harness)
        await ws.send(json.dumps({"type": "start", "session_id": "tail-session-1234", "audio_elapsed": 0.5}))
        await expect(ws, "started")
        for sequence, position in enumerate(range(0, 8000, 1600)):
            await ws.send(packet(sequence, position, 1600))
        await ws.send(json.dumps({"type": "stop", "session_id": "tail-session-1234"}))
        await ws.send(json.dumps({"type": "ping"}))
        await expect(ws, "pong")
        await ws.close()

    asyncio.run(deliver_tail())
    state = settled(harness)
    assert [r["frames"] for r in state["recordings"]] == [{"browser": 8000}]
    assert state["recordings"][0]["gaps"] == []
    # The harness transcribes no speech: the meeting fails readiness and stays in
    # the local capture; nothing unfinished reaches the synced folder.
    assert not state["recordings"][0]["published"]
    assert not list(harness.recordings.iterdir())


def test_a_start_while_saving_is_busy_not_closed_and_succeeds_once_idle(start_harness):
    harness = start_harness(finalize_delay=1.0)

    async def run():
        ws = await client(harness)
        await ws.send(json.dumps({"type": "start", "session_id": "first-meeting-123"}))
        await expect(ws, "started")
        await ws.send(packet(0, 0, 1600))
        await ws.send(json.dumps({"type": "stop", "session_id": "first-meeting-123"}))
        while harness.state()["phase"] == "recording":
            await asyncio.sleep(0.02)
        await ws.send(json.dumps({"type": "start", "session_id": "second-meeting-12"}))
        busy = await expect(ws, "error")
        assert (busy["code"], busy["session_id"]) == ("busy", "second-meeting-12")
        assert "second-meeting-12" not in harness.state()["closed_sessions"]
        await ws.send(json.dumps({"type": "ping"}))
        await expect(ws, "pong")  # the socket stays open for the terminal idle
        await expect(ws, "status", state="idle", session_id="first-meeting-123")
        while harness.state()["worker_alive"]:
            await asyncio.sleep(0.02)
        await ws.send(json.dumps({"type": "start", "session_id": "second-meeting-12"}))
        await expect(ws, "started", session_id="second-meeting-12")
        await ws.send(packet(0, 0, 1600))
        await ws.send(json.dumps({"type": "stop", "session_id": "second-meeting-12"}))
        await ws.close()

    asyncio.run(run())
    state = settled(harness)
    assert [r["frames"] for r in state["recordings"]] == [{"browser": 1600}, {"browser": 1600}]


def test_a_stop_sent_right_before_closing_is_honoured(start_harness):
    # The extension sends its last PCM and 'stop', then closes at once. Replies
    # to those packets fail on the closed peer; the stop must still be handled.
    harness = start_harness()

    async def run():
        ws = await client(harness)
        await ws.send(json.dumps({"type": "start", "session_id": "closing-session-1"}))
        await expect(ws, "started")
        for sequence, position in enumerate(range(0, 16000, 1600)):
            await ws.send(packet(sequence, position, 1600))
        await ws.send(json.dumps({"type": "stop", "session_id": "closing-session-1"}))
        await ws.close()

    asyncio.run(run())
    state = settled(harness)
    assert [r["frames"] for r in state["recordings"]] == [{"browser": 16000}]
