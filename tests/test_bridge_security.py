import struct
import numpy as np
import pytest
from whisper_daemon.bridge_protocol import is_allowed_origin, AudioSequence, validate_client_message

ID = 'a' * 32

def packet(seq, frame, samples):
    return struct.pack('<4sIQ', b'WDA1', seq, frame) + np.asarray(samples, dtype='<f4').tobytes()


def test_only_paired_exact_origin_allowed():
    assert is_allowed_origin('chrome-extension://' + ID, ID)
    for value in ('chrome-extension://', 'chrome-extension://' + 'b' * 32,
                  'chrome-extension://' + ID + '/path', None):
        assert not is_allowed_origin(value, ID)
    assert not is_allowed_origin('chrome-extension://' + ID)


def test_sequences_drop_duplicates_and_report_missing_samples():
    stream = AudioSequence()
    data, gap = stream.accept(packet(0, 0, [0.1, 0.2]))
    assert len(data) == 8 and gap == 0
    assert stream.accept(packet(0, 0, [0.1, 0.2])) is None
    data, gap = stream.accept(packet(2, 5, [0.3]))
    assert gap == 3 and stream.frames == 6


@pytest.mark.parametrize('payload', [b'x', packet(0, 0, [np.nan]), packet(0, 0, [np.inf]), packet(0, 0, [2]), packet(0, 0, [])])
def test_malformed_pcm_rejected(payload):
    with pytest.raises(ValueError): AudioSequence().accept(payload)


def test_control_messages_are_validated():
    for data in ([], {'type': 'start'}, {'type': 'start', 'session_id': '../oops'},
                 {'type': 'set_settings', 'patch': []}, {'type': 'unknown'}):
        with pytest.raises(ValueError): validate_client_message(data)


def test_real_websocket_auth_reconnect_gap_and_stop(tmp_path):
    import asyncio
    import json
    from websockets.asyncio.client import connect
    from websockets.asyncio.server import serve
    from websockets.exceptions import InvalidStatus, ConnectionClosed
    from whisper_daemon.audio_server import BrowserAudioBridge
    from whisper_daemon.pairing import pair
    paired = pair(ID, tmp_path / 'pair.json')
    received, starts, stops, gaps = [], [], [], []
    bridge = BrowserAudioBridge('127.0.0.1', 0,
        lambda *args: starts.append(args), lambda audio, position: received.append((position, audio)), lambda session: stops.append(session), gaps.append,
        pairing_path=tmp_path / 'pair.json')
    async def run():
        async with serve(bridge._handler, '127.0.0.1', 0, process_request=bridge._process_request) as server:
            url = f'ws://127.0.0.1:{server.sockets[0].getsockname()[1]}'
            with pytest.raises(InvalidStatus):
                async with connect(url, origin='chrome-extension://' + 'b' * 32): pass
            async with connect(url, origin='chrome-extension://' + ID) as bad:
                await bad.send(json.dumps({'type': 'auth', 'secret': 'wrong'}))
                with pytest.raises(ConnectionClosed): await bad.recv()
            assert not starts
            async def authenticate(ws):
                await ws.send(json.dumps({'type': 'auth', 'secret': paired['secret']}))
                assert json.loads(await ws.recv())['type'] == 'authenticated'
                await ws.send(json.dumps({'type': 'start', 'session_id': 'test-session-12345678'}))
                assert json.loads(await ws.recv())['type'] == 'started'
            async with connect(url, origin='chrome-extension://' + ID) as ws:
                await authenticate(ws)
                await ws.send(packet(0, 0, [.1, .2])); await ws.recv()
            assert not stops  # a transport reconnect is not a meeting stop
            async with connect(url, origin='chrome-extension://' + ID) as ws:
                await authenticate(ws)
                await ws.send(packet(0, 0, [.1, .2])); await ws.recv()
                await ws.send(packet(2, 5, [.3])); await ws.recv()
                await ws.send(json.dumps({'type': 'stop', 'session_id': 'test-session-12345678'}))
                await ws.send(json.dumps({'type': 'ping'})); await ws.recv()
            assert len(starts) == 1 and len(stops) == 1
            assert [position for position, _ in received] == [0, 5]
            assert len(b''.join(audio for _, audio in received)) == 3 * 4
            assert gaps[0]['samples'] == 3
            async with connect(url, origin='chrome-extension://' + ID) as ws:
                await ws.send(json.dumps({'type': 'auth', 'secret': paired['secret']})); await ws.recv()
                await ws.send('[]')
                assert json.loads(await ws.recv())['type'] == 'error'
    asyncio.run(run())
