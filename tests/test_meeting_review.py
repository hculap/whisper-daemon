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


def test_review_edit_of_a_ready_meeting_publishes_the_next_revision(tmp_path):
    import hashlib
    from whisper_daemon.meeting_manifest import begin_meeting, finalize_meeting
    make_transcript(tmp_path)
    prepare_review(tmp_path)
    first = finalize_meeting(tmp_path, begin_meeting(tmp_path, 'CRM'), None, [])
    edit = {'names': {'1': 'Anna'}}
    save_review(tmp_path, edit)
    second = json.loads((tmp_path / 'meeting.ready.json').read_text())
    assert second['revision'] == 2 and second['meeting_id'] == first['meeting_id']
    assert 'transcript_original.json' in {f['path'] for f in second['files']}
    for entry in second['files']:
        assert entry['sha256'] == hashlib.sha256((tmp_path / entry['path']).read_bytes()).hexdigest()
    save_review(tmp_path, edit)  # nothing changed: no new revision
    assert json.loads((tmp_path / 'meeting.ready.json').read_text())['revision'] == 2


def test_review_edit_never_marks_an_unfinished_meeting_ready(tmp_path):
    make_transcript(tmp_path)
    save_review(tmp_path, {'names': {'1': 'Anna'}})
    assert not (tmp_path / 'meeting.ready.json').exists()


def test_review_save_defers_memory_restart_without_blocking_watchdog(tmp_path, monkeypatch):
    import threading
    from whisper_daemon import meeting_review, supervisor

    make_transcript(tmp_path)
    prepare_review(tmp_path)
    original_atomic_json = meeting_review.atomic_json
    save_entered = threading.Event()
    allow_save = threading.Event()
    idle_probed = threading.Event()
    restarted = threading.Event()

    def blocking_atomic_json(path, data):
        if path.name == 'transcript.json':
            save_entered.set()
            assert allow_save.wait(2)
        return original_atomic_json(path, data)

    monkeypatch.setattr(meeting_review, 'atomic_json', blocking_atomic_json)
    monkeypatch.setattr(supervisor, 'request_restart', lambda reason: restarted.set())

    save_thread = threading.Thread(
        target=meeting_review.save_review,
        args=(tmp_path, {'names': {'1': 'Anna'}}),
    )
    save_thread.start()
    assert save_entered.wait(2)

    def is_idle():
        idle_probed.set()
        return True

    restart_thread = threading.Thread(
        target=supervisor.request_restart_if_safe,
        args=('test pressure', is_idle),
    )
    restart_thread.start()
    try:
        restart_thread.join(0.2)
        assert not restart_thread.is_alive()
        assert not idle_probed.wait(0.2)
        assert not restarted.is_set()
    finally:
        allow_save.set()
        save_thread.join(2)

    assert not save_thread.is_alive()
    assert supervisor.request_restart_if_safe('test pressure', is_idle)
    assert idle_probed.is_set()
    assert restarted.is_set()
