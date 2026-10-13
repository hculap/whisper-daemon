"""macOS synthetic speech -> real VAD/bridge/store/Whisper -> meeting exports.

Uses installed system voices and cached models; forbids external connections.
Does not touch the running daemon, user configuration, history or recordings.
"""
import asyncio
import json
import socket
import struct
import subprocess
import tempfile
import time
from pathlib import Path
from queue import Empty

import numpy as np
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve
from whisper_daemon.audio_server import BrowserAudioBridge
from whisper_daemon.formats import merge_chunk_results, FORMATTERS
from whisper_daemon.meeting_recorder import MeetingRecorder
from whisper_daemon.meeting_review import prepare_review
from whisper_daemon.pairing import pair
from whisper_daemon.recording_store import RecordingStore, DiskChunkQueue, atomic_json
from whisper_daemon.transcriber import transcribe_full

original_connect = socket.socket.connect
def local_only(sock, address):
    if isinstance(address, tuple) and address[0] in ('127.0.0.1', '::1'):
        return original_connect(sock, address)
    raise AssertionError('External network attempted during offline E2E')
socket.socket.connect = local_only


async def verify(path):
    source_audio = {}
    for lang, voice, sentence in [
        ('en', 'Samantha', 'We agreed to send the proposal on Friday. Anna will prepare the report. The final number is forty two.'),
        ('pl', 'Zosia', 'Ustaliliśmy, że wyślemy ofertę w piątek. Anna przygotuje raport. Ostatnia liczba to czterdzieści dwa.'),
    ]:
        file = path/f'{lang}.aiff'
        subprocess.run(['/usr/bin/say', '-v', voice, '-o', str(file), sentence], check=True)
        source_audio[lang] = np.frombuffer(subprocess.check_output([
            'ffmpeg', '-v', 'error', '-i', str(file), '-ar', '16000', '-ac', '1', '-f', 'f32le', 'pipe:1']), dtype='<f4').copy()
    store = RecordingStore(path/'meeting', clock=lambda: 0)
    jobs = DiskChunkQueue(path/'meeting')
    recorders = {}
    for source in ('mic', 'browser'):
        rec = MeetingRecorder(jobs, source_label=source,
            raw_sink=lambda audio, position, s=source: store.append(s, audio, at=position),
            sample_filter=store.unpaused_audio)
        rec.start_without_device()
        recorders[source] = rec
    recorders['mic'].feed_audio(source_audio['en'], position=0)
    recorders['mic'].stop_without_device()
    paired = pair('a'*32, path/'pairing.json')
    bridge = BrowserAudioBridge('127.0.0.1', 0, lambda *args: True,
        lambda audio, pos: recorders['browser'].feed_audio(np.frombuffer(audio, dtype='<f4'), position=pos),
        lambda session: recorders['browser'].stop_without_device(), pairing_path=path/'pairing.json')
    async with serve(bridge._handler, '127.0.0.1', 0, process_request=bridge._process_request) as server:
        url = f'ws://127.0.0.1:{server.sockets[0].getsockname()[1]}'
        async with connect(url, origin='chrome-extension://'+'a'*32) as ws:
            await ws.send(json.dumps({'type':'auth', 'secret':paired['secret']})); await ws.recv()
            await ws.send(json.dumps({'type':'start', 'session_id':'speech-e2e-session'})); await ws.recv()
            audio = source_audio['pl']
            for sequence, start in enumerate(range(0, len(audio), 1001)):
                await ws.send(struct.pack('<4sIQ', b'WDA1', sequence, 160000+start)+audio[start:start+1001].tobytes())
                await ws.recv()
            await ws.send(json.dumps({'type':'stop', 'session_id':'speech-e2e-session'}))
            await ws.send(json.dumps({'type':'ping'})); await ws.recv()
    store.finish()
    for source, lang, offset in [('mic', 'en', 0), ('browser', 'pl', 160000)]:
        raw = np.fromfile(store.path/f'{source}.f32', dtype='<f4')
        assert np.array_equal(raw[offset:], source_audio[lang])
    while True:
        try: chunk = jobs.get_nowait()
        except Empty: break
        language = 'en' if chunk.source == 'mic' else 'pl'
        result = transcribe_full(chunk.audio, language=language)
        jobs.ack(chunk, result)
    merged = merge_chunk_results(jobs.results())
    for source in ('mic', 'browser'):
        text = ' '.join(s['text'] for s in merged['segments'] if s['source'] == source)
        assert '42' in text, (source, text)
        print(json.dumps({'source':source, 'text':text}, ensure_ascii=False), flush=True)
    merged['duration'] = store.meta['duration']
    atomic_json(store.path/'transcript.json', merged)
    for fmt in ('txt', 'srt', 'vtt'):
        (store.path/f'transcript.{fmt}').write_text(FORMATTERS[fmt](merged))
    review = prepare_review(store.path)
    assert len(review['transcript']['speakers']) == 2
    assert review['notes']['tasks']
    return {'test':'offline_speech_e2e', 'seconds':store.meta['duration'],
        'segments':len(merged['segments']), 'languages':['en','pl'], 'final_number_preserved':True}

if __name__ == '__main__':
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='whisper-speech-e2e-') as directory:
        result = asyncio.run(verify(Path(directory)))
    result['runtime_seconds'] = round(time.monotonic()-started, 2)
    print(json.dumps(result))
