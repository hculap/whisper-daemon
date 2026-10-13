"""Accelerated 110-minute capture through real WebSocket/ingress/recorder/store.

No devices or inference: synthetic PCM and silent VAD. A reconnect replays ten
seconds at once while the microphone continues. Validates duration/content/RSS.
"""
import ast
import asyncio
import json
import logging
import resource
import struct
import tempfile
import threading
import time
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve
from whisper_daemon import meeting_recorder as mod
from whisper_daemon.audio_server import BrowserAudioBridge
from whisper_daemon.pairing import pair
from whisper_daemon.recording_store import RecordingStore, DiskChunkQueue

class SilentVAD:
    def reset_states(self): pass
    def __call__(self, samples): return 0
mod.SileroVAD = SilentVAD
logging.disable(logging.WARNING)


async def verify(path):
    now = [0.0]
    store = RecordingStore(path, clock=lambda: now[0])
    jobs = DiskChunkQueue(path)
    recorders = {}
    for source in ('mic', 'browser'):
        rec = mod.MeetingRecorder(jobs, source_label=source,
            raw_sink=lambda audio, position, s=source: store.append(s, audio, at=position),
            sample_filter=store.unpaused_audio)
        rec.start_without_device()
        recorders[source] = rec

    # Actual native ingress, without booting AppKit or the production daemon.
    tree = ast.parse(Path('src/whisper_daemon/menubar.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MenuBarDelegate')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_on_browser_audio')
    scope = {'np': np}
    exec(compile(ast.Module(body=[method], type_ignores=[]), 'menubar.py', 'exec'), scope)
    receiver = SimpleNamespace(_browser_audio_lock=threading.RLock(), _browser_recorder=recorders['browser'])
    identity = 'a'*32
    pairing = pair(identity, path/'pairing.json')
    starts = []
    bridge = BrowserAudioBridge('127.0.0.1', 0, lambda *args: starts.append(args),
        lambda data, position: scope['_on_browser_audio'](receiver, data, position), lambda session: None,
        pairing_path=path/'pairing.json')
    async with serve(bridge._handler, '127.0.0.1', 0, process_request=bridge._process_request) as server:
        url = f'ws://127.0.0.1:{server.sockets[0].getsockname()[1]}'
        async def attach():
            ws = await connect(url, origin='chrome-extension://'+identity, compression=None)
            await ws.send(json.dumps({'type':'auth', 'secret':pairing['secret']})); await ws.recv()
            await ws.send(json.dumps({'type':'start', 'session_id':'verification-110-minutes'})); await ws.recv()
            return ws
        ws = await attach()
        backlog = []
        baseline_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        for second in range(6600):
            now[0] = second+1
            block = np.full(16000, .001 if second%2 else 0, dtype='<f4')
            recorders['mic'].feed_audio(block, position=second*16000)
            packet = struct.pack('<4sIQ', b'WDA1', second, second*16000)+block.tobytes()
            if second == 2: await ws.close()
            if 2 <= second < 12: backlog.append(packet); continue
            if second == 12:
                ws = await attach()
                for delayed in backlog: await ws.send(delayed); await ws.recv()
                await ws.send(backlog[-1]); await ws.recv()  # lost-ACK replay
            await ws.send(packet); await ws.recv()
            if second%1200 == 0: print(json.dumps({'captured_minutes':second//60}), flush=True)
        await ws.send(json.dumps({'type':'stop', 'session_id':'verification-110-minutes'}))
        await ws.send(json.dumps({'type':'ping'})); await ws.recv(); await ws.close()
    assert len(starts) == 1
    for rec in recorders.values(): rec.stop_without_device()
    store.finish(duration=6600)
    assert store.meta['gaps'] == []
    total = 6600*16000
    for source in ('mic', 'browser', 'recording'):
        with wave.open(str(path/f'{source}.wav')) as wav: assert wav.getnframes() == total
    for source in ('mic', 'browser'):
        raw = np.memmap(path/f'{source}.f32', dtype='<f4', mode='r')
        assert len(raw) == total
        for second in (0, 1, 2, 3, 11, 12, 3599, 6598, 6599):
            assert np.all(raw[second*16000:(second+1)*16000] == (.001 if second%2 else 0))
        del raw
    rss_delta = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss-baseline_rss
    assert rss_delta < 180*1024*1024, rss_delta
    return {'test':'accelerated_110_min_websocket_two_sources', 'seconds_per_source':6600,
        'samples_per_source':total, 'replay_seconds':10, 'duplicate_replay':'ignored',
        'gaps':store.meta['gaps'], 'rss_growth_bytes':rss_delta}

if __name__ == '__main__':
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='whisper-110min-') as directory:
        report = asyncio.run(verify(Path(directory)))
    report['runtime_seconds'] = round(time.monotonic()-started, 2)
    print(json.dumps(report))
