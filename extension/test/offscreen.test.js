const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

function fixture() {
  const sockets = [], messages = [];
  let receive;
  class Socket {
    static OPEN = 1;
    constructor() { this.readyState = 0; this.sent = []; sockets.push(this); }
    open() { this.readyState = 1; this.onopen(); }
    close() { this.readyState = 3; }
    send(value) {
      this.sent.push(value);
      if (typeof value === 'string') {
        const data = JSON.parse(value);
        if (data.type === 'auth') queueMicrotask(() => this.onmessage({data: JSON.stringify({type:'authenticated'})}));
        if (data.type === 'start') queueMicrotask(() => this.onmessage({data: JSON.stringify({type:'started', session_id:data.session_id})}));
      }
    }
  }
  const context = vm.createContext({
    console, crypto: require('node:crypto').webcrypto, chrome: { storage: {local:{get:async () => ({pairingSecret:'test-secret'})}}, runtime: {
      onMessage: { addListener(fn) { receive = fn; } },
      sendMessage(msg) { messages.push(msg); return Promise.resolve(); },
    } }, WebSocket: Socket, Float32Array, Array, Date,
    setInterval() { return 1; }, clearInterval() {},
    setTimeout() { return 2; }, clearTimeout() {},
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../offscreen.js'), 'utf8'), context);
  return { run: code => vm.runInContext(code, context), sockets, messages,
    open: async () => { await new Promise(setImmediate); sockets[0].open(); },
    receive: msg => receive(msg, { tab: { id: 1 } }, () => {}) };
}

test('a broadcast PCM frame is ignored until addressed to offscreen', () => {
  const f = fixture();
  f.run('capturing = true; ws = { readyState: 1, send() { globalThis.sends = (globalThis.sends || 0) + 1; } }; globalThis.sends = 0');
  f.receive({ type: 'WD_PCM', samples: [0.1, 0.2] });
  f.receive({ type: 'WD_PCM', target: 'offscreen', samples: [0.1, 0.2] });
  assert.equal(f.run('sends'), 1);
});

test('stop cancels a connection that has not opened yet', async () => {
  const f = fixture();
  const pending = f.run('startExternal()');
  f.run('stopCapture()');
  await f.open();
  await pending;
  assert.equal(f.run('capturing'), false);
  assert.equal(f.sockets[0].sent.filter(x => JSON.parse(x).type === 'start').length, 0);
});

test('daemon-side stop permits a subsequent browser start', async () => {
  const f = fixture();
  const pending = f.run('startExternal()');
  await f.open();
  await pending;
  f.run('handleServerMessage({data: JSON.stringify({type: "status", state: "recording"})})');
  f.run('handleServerMessage({data: JSON.stringify({type: "status", state: "transcribing"})})');
  f.run('handleServerMessage({data: JSON.stringify({type: "status", state: "idle"})})');
  await f.run('startExternal()');
  assert.equal(f.sockets[0].sent.map(JSON.parse).filter(m => m.type === 'start').length, 2);
});

test('settings connection racing a start is shared without cancelling capture', async () => {
  const f = fixture();
  const control = f.run('ensureControlConnection()');
  const start = f.run('startExternal()');
  await f.open();
  assert.equal(f.sockets.length, 1);
  await Promise.all([control, start]);
  assert.equal(f.run('capturing'), true);
  assert.equal(f.sockets[0].sent.map(JSON.parse).filter(m => m.type === 'start').length, 1);
});
