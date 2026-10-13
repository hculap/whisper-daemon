import json
from whisper_daemon.meeting_review import prepare_review, save_review


def make_transcript(path):
    data = {'text': 'We agreed. I will send the proposal.', 'language': 'en', 'segments': [
        {'id': 0, 'start': 0, 'end': 2, 'source': 'mic', 'text': 'We agreed.'},
        {'id': 1, 'start': 4, 'end': 7, 'source': 'browser', 'text': 'I will send the proposal.'}]}
    (path / 'transcript.json').write_text(json.dumps(data))


def test_review_extracts_traceable_candidates_and_source_labels(tmp_path):
    make_transcript(tmp_path)
    data = prepare_review(tmp_path)
    assert data['transcript']['speakers'][1]['label'] == 'Participants'
    task = data['notes']['tasks'][0]
    assert task['start'] == 4 and task['confirmed'] is False
    assert task['text'] == 'I will send the proposal.'


def test_names_tasks_and_segment_assignment_persist_and_rebuild_exports(tmp_path):
    make_transcript(tmp_path)
    prepare_review(tmp_path)
    save_review(tmp_path, {'names': {'0': 'Szymon', '1': 'Anna'}, 'assignments': {'1': 0},
                           'summary': 'Agreed on the proposal.', 'tasks': [
        {'text': 'Send proposal', 'owner': 'Anna', 'due': 'Friday', 'start': 4, 'done': True, 'confirmed': True}]})
    data = prepare_review(tmp_path)
    assert data['transcript']['segments'][1]['speaker'] == 0
    assert data['transcript']['speakers'][0]['label'] == 'Szymon'
    assert data['notes']['tasks'][0]['done'] is True
    assert 'Szymon' in (tmp_path / 'transcript.txt').read_text()


def test_review_http_range_access_and_persisted_edit(tmp_path):
    import urllib.request
    import urllib.error
    import wave
    from whisper_daemon.meeting_review import ReviewServer
    make_transcript(tmp_path)
    with wave.open(str(tmp_path / 'recording.wav'), 'wb') as wav:
        wav.setparams((1, 2, 16000, 0, 'NONE', ''))
        wav.writeframes(bytes(16000 * 2 * 8))
    server = ReviewServer(tmp_path)
    url = server.start()
    try:
        request = urllib.request.Request(url + 'recording.wav', headers={'Range': 'bytes=44-75'})
        with urllib.request.urlopen(request) as response:
            assert response.status == 206 and len(response.read()) == 32
        for request in (url + '../transcript.json', url.replace(server.token, 'wrong') + 'data'):
            try: urllib.request.urlopen(request)
            except urllib.error.HTTPError as error: assert error.code in (403, 404)
            else: raise AssertionError('Private route accepted')
        origin = f'http://127.0.0.1:{server.server_port}'
        payload = json.dumps({'names': {'0': 'Szymon', '2': 'Nowa osoba'}, 'assignments': {'1': 2}}).encode()
        request = urllib.request.Request(url + 'save', data=payload,
            headers={'Origin': origin, 'Content-Type': 'application/json'})
        with urllib.request.urlopen(request) as response:
            assert json.loads(response.read())['transcript']['segments'][1]['speaker'] == 2
        request = urllib.request.Request(url + 'save', data=payload, headers={'Origin': 'https://evil.example'})
        try: urllib.request.urlopen(request)
        except urllib.error.HTTPError as error: assert error.code == 403
        else: raise AssertionError('Cross-origin mutation accepted')
    finally:
        server.shutdown(); server.server_close()
