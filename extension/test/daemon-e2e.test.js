// End-to-end: real background.js + offscreen.js over real WebSockets against the
// real bridge and meeting pipeline (tests/daemon_harness.py).
const test = require('node:test');
const assert = require('node:assert/strict');
const {available, setup, sleep, waitFor} = require('./support/e2e.js');

const skip = available ? false : 'project .venv not found (set WD_E2E_PYTHON)';
const timeout = 60000;
const MEETING = {title: 'Weekly sync', url: 'https://meet.google.com/abc-defg-hij'};

async function join(ext, id = 1) {
  const tab = ext.tab(id);
  await tab.send({type: 'MEET_JOINED', ...MEETING});
  return tab;
}

async function record(ext, tab, count = 1) {
  await tab.send({type: 'WD_START', ...MEETING});
  await waitFor(() => ext.sent('CAPTURE_STARTED').length >= count, {message: `CAPTURE_STARTED #${count}`});
}

const state = ext => ext.popup({type: 'GET_STATE'});
const recovery = ext => ext.popup({type: 'GET_RECOVERY', target: 'offscreen'});
// Idle with nothing queued: no live or pending session, every recording saved.
async function settled(daemon) {
  let last;
  try {
    return await waitFor(async () => {
      last = await daemon.state();
      return last.phase === 'idle' && !last.worker_alive && !last.start_pending && !last.session_id &&
        last.recordings.every(r => r.state === 'complete') && last;
    }, {timeout: 15000, message: 'daemon idle with complete recordings'});
  } catch (error) {
    error.message += `\n${JSON.stringify(last)}\n${daemon.stderr.join('').slice(-3000)}`;
    throw error;
  }
}

test('#1 a settings snapshot keeps the joined tab, so leaving the call closes the socket', {skip, timeout}, async t => {
  const {extension: ext, daemon} = await setup(t);
  const tab = await join(ext);
  await tab.send({type: 'WD_GET_SETTINGS'});
  await waitFor(() => ext.sent('STATUS').length, {message: 'snapshot status'});
  await waitFor(async () => (await daemon.state()).connected, {message: 'control socket'});

  await tab.send({type: 'MEET_LEFT'});
  await waitFor(async () => !(await daemon.state()).connected, {message: 'socket closed after leaving'});
  assert.ok(tab.inbox.some(m => m.type === 'WD_MEET_LEFT'));
  assert.equal((await state(ext)).state, 'idle');
});

test('#1 #8 a saved recording keeps the joined tab and has no phantom trailing gaps', {skip, timeout}, async t => {
  const {extension: ext, daemon} = await setup(t);
  const tab = await join(ext);
  await record(ext, tab);
  const samples = await tab.pcm(1.5, {realtime: true});
  await tab.send({type: 'WD_STOP'});
  const done = await settled(daemon);
  await waitFor(() => ext.sent('STATUS').some(m => m.state === 'idle'), {message: 'idle relayed'});

  assert.equal(done.recordings.length, 1);
  assert.equal(done.recordings[0].frames.browser, samples);
  assert.deepEqual(done.recordings[0].gaps, []);
  assert.deepEqual(await state(ext), {state: 'detected', meetTabId: 1, meetTitle: MEETING.title, meetUrl: MEETING.url, connection: 'connected'});

  await tab.send({type: 'MEET_LEFT'});
  await waitFor(async () => !(await daemon.state()).connected, {message: 'socket closed after leaving'});
  assert.ok(tab.inbox.some(m => m.type === 'WD_MEET_LEFT'));
});

test('#2 a retained tail reaches a restarted daemon and the next recording waits for it to be saved', {skip, timeout}, async t => {
  const env = await setup(t);
  const ext = env.extension;
  const tab = await join(ext);
  await record(ext, tab);
  await tab.pcm(0.5);
  await waitFor(() => ext.delivered(), {message: 'first half acknowledged'});
  await env.daemon.kill();
  await tab.pcm(0.5);
  await tab.send({type: 'WD_STOP'});
  await waitFor(() => ext.sent('CAPTURE_ERROR').length, {message: 'tail retained'});
  assert.equal((await recovery(ext)).pending, true);

  // A busy main thread lets the bridge see the tail's stop before the meeting starts.
  const daemon = await env.restartDaemon({finalizeDelay: 1.5, mainLatency: 0.2});
  const errorsBefore = ext.sent('CAPTURE_ERROR').length;
  const startedAt = Date.now();
  await tab.send({type: 'WD_START', ...MEETING});
  const live = await tab.pcm(1, {realtime: true});
  await waitFor(() => ext.sent('CAPTURE_STARTED').length >= 2, {timeout: 9000, message: 'second recording started'});
  assert.ok(Date.now() - startedAt < 9000);
  await tab.pcm(0.5);
  await tab.send({type: 'WD_STOP'});

  const done = await settled(daemon);
  assert.deepEqual(ext.sent('CAPTURE_ERROR').slice(errorsBefore), []);
  assert.deepEqual(await recovery(ext), {count: 0, pending: false});
  const [tail, next] = done.recordings;
  assert.equal(done.recordings.length, 2, JSON.stringify(done.recordings));
  assert.equal(tail.frames.browser, 16000, 'retained tail stored at its original position');
  assert.deepEqual(tail.gaps.map(g => [g.start, g.end, g.reason]), [[0, 0.5, 'missing audio']]);
  assert.equal(next.frames.browser, live + 8000, 'audio buffered while waiting is not lost');
  assert.deepEqual(next.gaps, []);
});

test('#4 stopping a start the daemon refused leaves no phantom recovery tail', {skip, timeout}, async t => {
  const {extension: ext, daemon} = await setup(t, {finalizeDelay: 2});
  const tab = await join(ext);
  await record(ext, tab);
  await tab.pcm(0.3);
  await tab.send({type: 'WD_STOP'});
  await tab.send({type: 'WD_START', ...MEETING});
  await sleep(300);
  await tab.pcm(0.2);
  await tab.send({type: 'WD_STOP'});
  await sleep(1000);

  assert.deepEqual(ext.sent('RECOVERY_AVAILABLE'), []);
  assert.deepEqual(await recovery(ext), {count: 0, pending: false});
  const done = await settled(daemon);
  assert.equal(done.closed_sessions.length, 1, 'only the real meeting is closed');
  assert.equal(done.recordings.length, 1);
});

test('#3 an unexported recovery tail does not block the next recording', {skip, timeout}, async t => {
  // 2 s live budget; the retained tail alone is ~1.9 s.
  const {extension: ext, daemon} = await setup(t, {overrides: {MAX_BUFFERED_BYTES: 128000}});
  const tab = await join(ext);
  await record(ext, tab);
  await tab.pcm(0.2);
  await waitFor(() => ext.delivered(), {message: 'first audio acknowledged'});
  await daemon.request({cmd: 'block', on: true});
  await tab.pcm(1.9);
  await daemon.request({cmd: 'native_stop'});
  await tab.send({type: 'WD_STOP'});
  await waitFor(() => ext.sent('CAPTURE_ERROR').length, {message: 'tail retained while unreachable'});
  await daemon.request({cmd: 'block', on: false});
  await settled(daemon);

  await tab.send({type: 'WD_START', ...MEETING});
  await waitFor(() => ext.sent('RECOVERY_AVAILABLE').length, {message: 'tail kept for export'});
  await waitFor(() => ext.sent('CAPTURE_STARTED').length >= 2, {message: 'next recording started'});
  const samples = await tab.pcm(1, {realtime: true});
  await tab.send({type: 'WD_STOP'});
  const done = await settled(daemon);

  assert.ok(!ext.sent('CAPTURE_ERROR').some(m => /Bufor połączenia audio jest pełny/.test(m.message)), JSON.stringify(ext.sent('CAPTURE_ERROR')));
  assert.equal(done.recordings.at(-1).frames.browser, samples);
  assert.equal((await recovery(ext)).count, 1);
});

test('#5 a daemon crash while saving does not block the next start', {skip, timeout}, async t => {
  const env = await setup(t, {transcribeDelay: 30});
  const ext = env.extension;
  const tab = await join(ext);
  await record(ext, tab);
  await tab.pcm(0.3);
  await tab.send({type: 'WD_STOP'});
  await waitFor(async () => (await state(ext)).state === 'transcribing', {message: 'saving'});
  await env.daemon.kill();
  await env.restartDaemon();

  await waitFor(async () => (await state(ext)).state !== 'transcribing', {message: 'saving state released'});
  const response = await tab.send({type: 'WD_START', ...MEETING});
  assert.equal(response.ok, true, JSON.stringify(response));
  await waitFor(() => ext.sent('CAPTURE_STARTED').length >= 2, {message: 'next recording started'});
});

test('#6 leaving during a reconnect delivers the tail, reports stopped and closes the socket', {skip, timeout}, async t => {
  const {extension: ext, daemon} = await setup(t);
  const tab = await join(ext);
  await record(ext, tab);
  await tab.pcm(0.3);
  await daemon.request({cmd: 'block', on: true});
  await tab.pcm(0.3);
  await tab.send({type: 'MEET_LEFT'});
  await sleep(150);
  await daemon.request({cmd: 'block', on: false});

  await waitFor(() => ext.sent('CAPTURE_STOPPED').length, {message: 'CAPTURE_STOPPED'});
  const done = await settled(daemon);
  assert.equal(done.recordings[0].frames.browser, tab.position);
  await waitFor(async () => !(await daemon.state()).connected, {message: 'socket closed'});
  assert.equal(ext.openSockets(), 0);
});

test('#7 a popup start with a retained tail always settles the starting state', {skip, timeout}, async t => {
  const env = await setup(t);
  const ext = env.extension;
  const tab = await join(ext);
  await record(ext, tab);
  await tab.pcm(0.2);
  await env.daemon.kill();
  await tab.pcm(0.2);
  await tab.send({type: 'WD_STOP'});
  await waitFor(() => ext.sent('CAPTURE_ERROR').length, {message: 'tail retained'});
  const daemon = await env.restartDaemon();

  await ext.popup({type: 'START_WITH_STREAM', streamId: 'stream-1', tabId: 1});
  await tab.send({type: 'WD_GET_SETTINGS'});
  await waitFor(() => ext.sent('CAPTURE_STARTED').length >= 2, {message: 'popup capture started'});
  assert.equal((await state(ext)).state, 'capturing');
  ext.media.worklet.port.onmessage({data: {samples: new Float32Array(1600).fill(0.25), position: 0}});
  await waitFor(async () => (await daemon.state()).recordings.length === 2, {message: 'popup meeting recorded'});
  await ext.popup({type: 'STOP_RECORDING'});
  await waitFor(() => ext.sent('CAPTURE_STOPPED').length, {message: 'popup capture stopped'});
  const done = await settled(daemon);
  assert.equal(done.recordings.at(-1).frames.browser, 1600);
});

test('leaving the call mid-recording stops and saves the meeting with all its audio', {skip, timeout}, async t => {
  const {extension: ext, daemon} = await setup(t);
  const tab = await join(ext);
  await record(ext, tab);
  const samples = await tab.pcm(1);
  await tab.send({type: 'MEET_LEFT'});
  await waitFor(() => ext.sent('CAPTURE_STOPPED').length, {message: 'CAPTURE_STOPPED'});
  const done = await settled(daemon);
  assert.equal(done.recordings[0].frames.browser, samples);
  assert.equal(ext.openSockets(), 0);
});
