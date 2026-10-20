const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

function fixture({autoStart = true, secret = 'test-secret', acceptKey = true} = {}) {
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
        if (data.type === 'auth' && acceptKey) queueMicrotask(() => this.onmessage({data: JSON.stringify({type:'authenticated'})}));
        if (data.type === 'start' && autoStart) queueMicrotask(() => this.onmessage({data: JSON.stringify({type:'started', session_id:data.session_id})}));
      }
    }
  }
  const context = vm.createContext({
    // Like Chrome, an offscreen document has chrome.runtime only: no storage.
    console, crypto: require('node:crypto').webcrypto, chrome: { runtime: {
      onMessage: { addListener(fn) { receive = fn; } },
      sendMessage(msg) {
        // The service worker answers pairing reads for this document.
        if (msg.type === 'GET_PAIRING_SECRET') return Promise.resolve({ secret });
        messages.push(msg); return Promise.resolve();
      },
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

test('capture acknowledgement includes audio buffered before the daemon start ACK', async () => {
  const f = fixture({autoStart:false});
  const pending = f.run('startExternal()');
  f.receive({type:'WD_PCM',target:'offscreen',samples:new Array(16000).fill(.1),position:0});
  await f.open(); await new Promise(setImmediate);
  f.run('handleServerMessage({data:JSON.stringify({type:"started",session_id:sessionId})})');
  await pending;
  assert.equal(f.messages.find(m=>m.type==='CAPTURE_STARTED').elapsed,1);
});

test('reconnect acknowledgement reports the existing capture position', async () => {
  const f = fixture({autoStart:false});
  const start=f.run('startExternal()'); await f.open(); await new Promise(setImmediate);
  f.run('handleServerMessage({data:JSON.stringify({type:"started",session_id:sessionId})})'); await start;
  f.messages.length=0;
  f.receive({type:'WD_PCM',target:'offscreen',samples:new Array(8000).fill(.1),position:0});
  f.sockets[0].readyState=3; f.sockets[0].onclose();
  await new Promise(setImmediate); f.sockets[1].open(); await new Promise(setImmediate);
  f.run('handleServerMessage({data:JSON.stringify({type:"started",session_id:sessionId})})');
  await new Promise(setImmediate);
  assert.equal(f.messages.find(m=>m.type==='CAPTURE_STARTED').elapsed,.5);
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

test('a late status for a stopped session cannot cancel the teardown awaiting its drain', async () => {
  const f=fixture({autoStart:false});
  const start=f.run('startExternal()');await f.open();await new Promise(setImmediate);
  f.run('handleServerMessage({data:JSON.stringify({type:"started",session_id:sessionId})})');await start;
  f.receive({type:'WD_PCM',target:'offscreen',samples:[.1,.2],position:0});
  f.sockets[0].readyState=3;f.sockets[0].onclose();
  const stop=f.run('stopCapture()');
  await new Promise(setImmediate);
  // Before the drain reconnects, the daemon reports the same session saving.
  f.run('handleServerMessage({data:JSON.stringify({type:"status",state:"transcribing",session_id:sessionId})})');
  f.sockets[1].open();await new Promise(setImmediate);
  f.run('handleServerMessage({data:JSON.stringify({type:"started",session_id:sessionId})})');
  await stop;
  assert.equal(JSON.parse(f.sockets[1].sent.at(-1)).type,'stop');
  assert.equal(f.messages.filter(m=>m.type==='CAPTURE_STOPPED').length,1);
  assert.equal(f.sockets[1].readyState,3);
});

test('a popup start superseded while the retained tail drains still settles background', async () => {
  const f=fixture({autoStart:false});
  f.run('stoppedSession={id:"old-session-123456",position:2,packets:[new ArrayBuffer(24)]}');
  const start=f.run('startCaptureFromStream("stream-1")');
  await f.open();await new Promise(setImmediate);
  f.receive({type:'WD_STOP',target:'offscreen'});
  f.run('handleServerMessage({data:JSON.stringify({type:"started",session_id:"old-session-123456"})})');
  await start;
  assert.ok(f.messages.some(m=>['CAPTURE_STOPPED','CAPTURE_ERROR','CAPTURE_STARTED'].includes(m.type)));
  assert.equal(f.run('sessionStarting'),false);
});

test('a busy daemon makes the start wait with audio buffered, then start once idle', async () => {
  const f=fixture({autoStart:false});
  const start=f.run('startExternal()');await f.open();await new Promise(setImmediate);
  const id=f.run('sessionId');
  f.run(`handleServerMessage({data:JSON.stringify({type:"error",code:"busy",session_id:"${id}"})})`);
  assert.equal(f.run('startSent'),false); // the daemon does not own it: a stop leaves no tail
  f.receive({type:'WD_PCM',target:'offscreen',samples:[.1,.2],position:0});
  f.run('handleServerMessage({data:JSON.stringify({type:"status",state:"idle",session_id:"previous-meeting-1"})})');
  f.run(`handleServerMessage({data:JSON.stringify({type:"started",session_id:"${id}"})})`);
  await start;
  const sent=f.sockets[0].sent.map(x=>typeof x==='string'?JSON.parse(x).type:'pcm');
  assert.deepEqual(sent.filter(x=>x!=='auth'),['start','start','pcm']);
  assert.equal(f.run('capturing'),true);
  assert.equal(f.messages.some(m=>m.type==='CAPTURE_ERROR'),false);
});

test('a retained recovery tail does not count against the live audio budget', async () => {
  const f=fixture();
  f.run('recoveryTails.push({id:"t",position:0,packets:[new ArrayBuffer(MAX_BUFFERED_BYTES)]})');
  const start=f.run('startExternal()');await f.open();await start;
  f.receive({type:'WD_PCM',target:'offscreen',samples:[.1,.2],position:0});
  assert.equal(f.run('capturing'),true);
  assert.equal(f.messages.some(m=>m.type==='CAPTURE_ERROR'),false);
});

const connectionStates = f => f.messages.filter(m => m.type === 'CONNECTION').map(m => m.state);

test('a settings read without a pairing key reports unpaired, not a capture error', async () => {
  const f = fixture({secret: null});
  await f.run('sendControl({type: "get_settings"})');
  assert.deepEqual(connectionStates(f), ['unpaired']);
  assert.equal(f.sockets.length, 0);
  assert.equal(f.messages.some(m => m.type === 'CAPTURE_ERROR'), false);
});

test('an unreachable daemon reports offline, not a capture error', async () => {
  const f = fixture();
  const pending = f.run('sendControl({type: "get_settings"})');
  await new Promise(setImmediate);
  f.sockets[0].onerror(new Error('refused'));
  f.sockets[0].onclose();
  await pending;
  assert.deepEqual(connectionStates(f), ['offline']);
  assert.equal(f.messages.some(m => m.type === 'CAPTURE_ERROR'), false);
});

test('a daemon that closes the socket during auth reports a rejected key', async () => {
  const f = fixture({acceptKey: false});
  const pending = f.run('sendControl({type: "get_settings"})');
  await f.open();
  f.sockets[0].onclose();
  await pending;
  assert.deepEqual(connectionStates(f), ['rejected']);
});

test('an authenticated control connection reports connected', async () => {
  const f = fixture();
  const pending = f.run('sendControl({type: "get_settings"})');
  await f.open();
  await pending;
  assert.deepEqual(connectionStates(f), ['connected']);
  assert.ok(f.sockets[0].sent.some(x => typeof x === 'string' && JSON.parse(x).type === 'get_settings'));
});

test('a start without a pairing key asks to pair instead of restarting the daemon', async () => {
  const f = fixture({secret: null});
  await f.run('startExternal()');
  const errors = f.messages.filter(m => m.type === 'CAPTURE_ERROR');
  assert.equal(errors.length, 1);
  assert.match(errors[0].message, /Sparuj rozszerzenie/);
});
