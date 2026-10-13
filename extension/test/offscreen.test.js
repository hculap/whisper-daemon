const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

function fixture({autoStart = true} = {}) {
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
        if (data.type === 'start' && autoStart) queueMicrotask(() => this.onmessage({data: JSON.stringify({type:'started', session_id:data.session_id})}));
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
  f.run('capturing = true; streamReady = true; ws = { readyState: 1, send() { globalThis.sends = (globalThis.sends || 0) + 1; } }; globalThis.sends = 0');
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
  f.run('handleServerMessage({data: JSON.stringify({type: "status", state: "transcribing", session_id:sessionId})})');
  f.run('handleServerMessage({data: JSON.stringify({type: "status", state: "idle", session_id:sessionId})})');
  await f.run('startExternal()');
  assert.equal(f.sockets[0].sent.map(JSON.parse).filter(m => m.type === 'start').length, 2);
});

test('reconnect never sends live PCM ahead of unacknowledged backlog', async () => {
  const f = fixture({autoStart:false});
  const start = f.run('startExternal()'); await f.open(); await new Promise(setImmediate);
  f.run('handleServerMessage({data:JSON.stringify({type:"started",session_id:sessionId})})'); await start;
  f.receive({type:'WD_PCM',target:'offscreen',samples:[.1,.2],position:0});
  f.run('handleServerMessage({data:JSON.stringify({type:"audio_ack",session_id:sessionId,sequence:0})})');
  f.sockets[0].readyState=3; f.sockets[0].onclose();
  f.receive({type:'WD_PCM',target:'offscreen',samples:[.3,.4],position:2});
  await new Promise(setImmediate); f.sockets[1].open(); await new Promise(setImmediate);
  f.receive({type:'WD_PCM',target:'offscreen',samples:[.5,.6],position:4});
  f.run('handleServerMessage({data:JSON.stringify({type:"started",session_id:sessionId})})');
  await new Promise(setImmediate);
  const packets=f.sockets[1].sent.filter(x=>typeof x!=='string');
  assert.deepEqual(packets.map(x=>new DataView(x).getUint32(4,true)),[1,2]);
});

test('stop after start was sent but before ACK explicitly terminates session', async () => {
  const f=fixture({autoStart:false});
  const start=f.run('startExternal()'); await f.open(); await new Promise(setImmediate);
  f.run('stopCapture()');
  await new Promise(setImmediate);
  f.run('handleServerMessage({data:JSON.stringify({type:"started",session_id:sessionId})})');
  await new Promise(setImmediate);
  assert.equal(f.sockets[0].sent.map(JSON.parse).filter(m=>m.type==='stop').length,1);
  f.run('handleServerMessage({data:JSON.stringify({type:"started",session_id:sessionId})})');
  await start;
  assert.equal(f.run('capturing'),false);
});

test('stop during reconnect drains pending PCM before stopping and shares the socket',async()=>{
  const f=fixture({autoStart:false});const start=f.run('startExternal()');
  await f.open();await new Promise(setImmediate);
  f.run('handleServerMessage({data:JSON.stringify({type:"started",session_id:sessionId})})');await start;
  f.receive({type:'WD_PCM',target:'offscreen',samples:[.1,.2],position:0});
  f.run('handleServerMessage({data:JSON.stringify({type:"audio_ack",session_id:sessionId,sequence:0})})');
  f.sockets[0].readyState=3;f.sockets[0].onclose();
  f.receive({type:'WD_PCM',target:'offscreen',samples:[.3,.4],position:2});
  const stop=f.run('stopCapture()');
  await new Promise(setImmediate);
  assert.equal(f.sockets.length,2);
  f.sockets[1].open();await new Promise(setImmediate);
  f.run('handleServerMessage({data:JSON.stringify({type:"started",session_id:sessionId})})');
  await stop;
  const sent=f.sockets[1].sent;
  const packets=sent.filter(x=>typeof x!=='string');
  assert.deepEqual(packets.map(x=>new DataView(x).getUint32(4,true)),[1]);
  assert.equal(JSON.parse(sent.at(-1)).type,'stop');
});

test('an idle snapshot cannot terminate a new session awaiting its ACK', async () => {
  const f=fixture({autoStart:false});
  const start=f.run('startExternal()'); await f.open(); await new Promise(setImmediate);
  f.run('handleServerMessage({data:JSON.stringify({type:"status",state:"idle",snapshot:true})})');
  f.run('handleServerMessage({data:JSON.stringify({type:"started",session_id:sessionId})})');
  await start;
  assert.equal(f.run('capturing'),true);
  assert.equal(f.messages.filter(m=>m.type==='STATUS').length,0);
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

test('native-ended session keeps its tail for export and allows a new capture',async()=>{
  const f=fixture({autoStart:false});
  const start=f.run('startExternal()');await f.open();await new Promise(setImmediate);
  f.run('handleServerMessage({data:JSON.stringify({type:"started",session_id:sessionId})})');await start;
  const oldId=f.run('sessionId');
  f.run('stoppedSession={id:sessionId,position:2,packets:[new ArrayBuffer(24)]};capturing=false;streamReady=false;');
  const next=f.run('startExternal()');await new Promise(setImmediate);
  f.receive({type:'WD_PCM',target:'offscreen',samples:[.7,.8],position:0});
  f.run('handleServerMessage({data:JSON.stringify({type:"error",code:"session_ended",session_id:stoppedSession.id,message:"ended"})})');
  await new Promise(setImmediate);
  f.sockets[1].open();await new Promise(setImmediate);
  f.run('handleServerMessage({data:JSON.stringify({type:"started",session_id:sessionId})})');await next;
  assert.notEqual(f.run('sessionId'),oldId);
  assert.equal(f.run('capturing'),true);
  assert.equal(f.run('recoveryTails[0].packets.length'),1);
  assert.equal(f.messages.some(m=>m.type==='CAPTURE_ERROR'),false);
  assert.equal(f.sockets[1].sent.filter(m=>typeof m!=='string').length,1);
});
