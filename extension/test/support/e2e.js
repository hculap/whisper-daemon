// End-to-end support: the real background.js + offscreen.js talking over real
// WebSockets to tests/daemon_harness.py (real bridge + real meeting pipeline).
const assert = require('node:assert/strict');
const {spawn} = require('node:child_process');
const fs = require('node:fs');
const net = require('node:net');
const os = require('node:os');
const path = require('node:path');
const readline = require('node:readline');
const vm = require('node:vm');

const ROOT = path.join(__dirname, '../../..');
const EXTENSION_DIR = path.join(ROOT, 'extension');
const PYTHON = process.env.WD_E2E_PYTHON || path.join(ROOT, '.venv/bin/python');
const HARNESS = path.join(ROOT, 'tests/daemon_harness.py');
const VERBOSE = !!process.env.WD_E2E_VERBOSE;
const available = fs.existsSync(PYTHON);

// Faster reconnects for tests. The start-ACK timeout literal is left as is so
// a test can prove a rejection arrives long before it.
const DEFAULT_OVERRIDES = {CONNECT_DELAY_MS: 50, RECONNECT_INTERVAL_MS: 100, RECONNECT_WINDOW_MS: 3000};

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

async function waitFor(check, {timeout = 8000, interval = 25, message = 'condition'} = {}) {
  const deadline = Date.now() + timeout;
  let last;
  while (Date.now() < deadline) {
    last = await check();
    if (last) return last;
    await sleep(interval);
  }
  throw new Error(`Timed out waiting for ${message}`);
}

function freePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.once('error', reject);
    server.listen(0, '127.0.0.1', () => {
      const {port} = server.address();
      server.close(() => resolve(port));
    });
  });
}

// options: {finalizeDelay, transcribeDelay, mainLatency} in seconds -> harness flags.
async function startDaemon({port, workdir, pairing, ...options}) {
  const flags = Object.entries(options).flatMap(([key, value]) =>
    [`--${key.replace(/[A-Z]/g, c => `-${c.toLowerCase()}`)}`, String(value)]);
  const child = spawn(PYTHON, [HARNESS, '--port', String(port), '--workdir', workdir, '--pairing', pairing, ...flags],
    {cwd: ROOT, stdio: ['pipe', 'pipe', 'pipe']});
  const pending = [];
  const stderr = [];
  let ready;
  const readyPromise = new Promise((resolve, reject) => {
    ready = resolve;
    child.once('exit', code => reject(new Error(`daemon harness exited ${code}:\n${stderr.join('')}`)));
  });
  child.stderr.on('data', chunk => { stderr.push(String(chunk)); if (VERBOSE) process.stderr.write(chunk); });
  readline.createInterface({input: child.stdout}).on('line', line => {
    const event = JSON.parse(line);
    if (event.event === 'ready') ready(event);
    else pending.shift()?.(event);
  });
  const exited = new Promise(resolve => child.once('exit', resolve));
  const info = await readyPromise;
  const request = command => new Promise(resolve => {
    pending.push(resolve);
    child.stdin.write(JSON.stringify(command) + '\n');
  });
  return {
    ...info,
    stderr,
    state: () => request({cmd: 'state'}),
    request,
    async kill() { child.kill('SIGKILL'); await exited; },
    async stop() {
      if (child.exitCode === null && child.signalCode === null) {
        await request({cmd: 'shutdown'});
        child.stdin.end();
      }
      await exited;
    },
  };
}

function loadScript(context, file, overrides) {
  let source = fs.readFileSync(path.join(EXTENSION_DIR, file), 'utf8');
  for (const [name, value] of Object.entries(overrides)) {
    const pattern = new RegExp(`const ${name} = [^;]+;`);
    assert.match(source, pattern, `override target ${name} missing in ${file}`);
    source = source.replace(pattern, `const ${name} = ${value};`);
  }
  vm.runInContext(source, context, {filename: file});
}

// Media fakes for the legacy popup capture path (getUserMedia + worklet).
function mediaFakes() {
  const media = {worklet: null, streams: 0};
  class AudioWorkletNode {
    constructor() { this.port = {onmessage: null}; media.worklet = this; }
    connect() {}
  }
  class AudioContext {
    constructor() { this.audioWorklet = {addModule: async () => {}}; this.destination = {}; }
    createMediaStreamSource() { return {connect() {}}; }
    close() { return Promise.resolve(); }
  }
  const track = {stop() {}, onended: null};
  const navigator = {mediaDevices: {getUserMedia: async () => {
    media.streams++;
    return {getAudioTracks: () => [track], getTracks: () => [track]};
  }}};
  return {media, globals: {AudioContext, AudioWorkletNode, navigator}};
}

function createExtension({port, secret, extensionId, overrides = {}}) {
  const listeners = {};
  const inboxes = new Map();
  const events = [];
  const notifications = [];
  const logs = [];
  const sockets = [];
  const tabs = new Set();
  const handlers = {};
  const clone = value => JSON.parse(JSON.stringify(value));

  // chrome.runtime.sendMessage: delivered asynchronously to every other
  // extension context; a content script's message reaches both pages.
  function dispatch(from, msg, sender) {
    events.push({from, msg: clone(msg)});
    return new Promise(resolve => setImmediate(() => {
      let settled = false;
      // Responses cross the vm realm boundary; clone them like Chrome serializes them.
      const respond = value => { if (!settled) { settled = true; resolve(value === undefined ? value : clone(value)); } };
      for (const name of ['background', 'offscreen']) {
        if (name !== from && listeners[name]) listeners[name](clone(msg), sender, respond);
      }
      setTimeout(() => respond(undefined), 50);
    }));
  }

  const quietConsole = Object.fromEntries(['log', 'info', 'debug', 'warn', 'error'].map(level =>
    [level, (...args) => { logs.push([level, ...args.map(String)].join(' ')); if (VERBOSE) console.error(`[ext ${level}]`, ...args); }]));

  class ExtensionSocket extends WebSocket {
    constructor() {
      super(`ws://127.0.0.1:${port}`, {headers: {Origin: `chrome-extension://${extensionId}`}});
      sockets.push(this);
    }
  }

  const pageUrl = page => `chrome-extension://${extensionId}/${page}`;

  // Like Chrome: an offscreen document gets chrome.runtime only, no storage.
  function chromeFor(name) {
    const storage = name === 'offscreen' ? {} :
      {storage: {local: {setAccessLevel() {}, get: async () => ({pairingSecret: secret})}}};
    return {
      ...storage,
      runtime: {
        id: extensionId,
        getURL: pageUrl,
        onMessage: {addListener(fn) { listeners[name] = fn; }},
        sendMessage: msg => dispatch(name, msg, {id: extensionId, url: pageUrl(`${name}.html`)}),
        getContexts: async () => [{}],
      },
      offscreen: {createDocument: async () => {}},
      action: {setBadgeText() {}, setBadgeBackgroundColor() {}, openPopup() {}},
      notifications: {
        create(id, options) { notifications.push({id, ...options}); },
        clear() {},
        onClicked: {addListener(fn) { handlers.notificationClick = fn; }},
      },
      tabs: {
        sendMessage: async (id, msg) => {
          if (!inboxes.has(id)) inboxes.set(id, []);
          inboxes.get(id).push(clone(msg));
        },
        query: async () => [...tabs].map(id => ({id, active: true})),
        update: async () => {},
        onRemoved: {addListener(fn) { handlers.tabRemoved = fn; }},
      },
    };
  }

  const timers = {setTimeout, clearTimeout, setInterval, clearInterval, setImmediate, queueMicrotask};
  const binary = {ArrayBuffer, Float32Array, Uint8Array, DataView, Blob, URL};
  const background = vm.createContext({chrome: chromeFor('background'), console: quietConsole, ...timers});
  loadScript(background, 'background.js', {});
  const {media, globals} = mediaFakes();
  const offscreen = vm.createContext({chrome: chromeFor('offscreen'), console: quietConsole,
    crypto: require('node:crypto').webcrypto, WebSocket: ExtensionSocket, ...timers, ...binary, ...globals});
  loadScript(offscreen, 'recovery.js', {});
  loadScript(offscreen, 'offscreen.js', {...DEFAULT_OVERRIDES, ...overrides});

  function tab(id) {
    tabs.add(id);
    if (!inboxes.has(id)) inboxes.set(id, []);
    let position = 0;
    return {
      id,
      inbox: inboxes.get(id),
      send(msg) {
        if (msg.type === 'WD_START') position = 0; // wd-content restarts its sample clock
        return dispatch('tab', msg, {tab: {id}});
      },
      // Real-time paced or burst PCM, as the page's worklet relays it.
      async pcm(seconds, {frame = 1600, realtime = false} = {}) {
        const total = Math.round(seconds * 16000);
        for (let sent = 0; sent < total; sent += frame) {
          const size = Math.min(frame, total - sent);
          const samples = Array.from({length: size}, (_, i) => ((position + i) % 200) / 1000);
          await dispatch('tab', {type: 'WD_PCM', samples, position}, {tab: {id}});
          position += size;
          if (realtime) await sleep(size / 16);
        }
        return position;
      },
      get position() { return position; },
    };
  }

  return {
    tab, media, events, notifications, logs, sockets,
    popup: msg => dispatch('popup', msg, {}),
    background: code => vm.runInContext(code, background),
    offscreen: code => vm.runInContext(code, offscreen),
    sent: type => events.filter(e => e.msg.type === type).map(e => e.msg),
    openSockets: () => sockets.filter(s => s.readyState === WebSocket.OPEN).length,
    // The daemon ACKed every packet sent so far (the offscreen replay buffer is empty).
    delivered: () => vm.runInContext('pcmBuffer.length === 0 && capturing', offscreen),
    closeTab(id) { tabs.delete(id); handlers.tabRemoved?.(id); },
    async shutdown() {
      vm.runInContext('cleanup()', offscreen);
      await sleep(20);
    },
  };
}

// daemonOptions: harness timing (see startDaemon); overrides: offscreen.js constants.
async function setup(t, {overrides = {}, ...daemonOptions} = {}) {
  const port = await freePort();
  const workdir = fs.mkdtempSync(path.join(os.tmpdir(), 'wd-e2e-'));
  const pairing = path.join(workdir, 'pairing.json');
  const daemons = [];
  const launch = async options => {
    const daemon = await startDaemon({port, pairing, ...options, workdir: path.join(workdir, `daemon-${daemons.length + 1}`)});
    daemons.push(daemon);
    return daemon;
  };
  await launch(daemonOptions);
  const first = daemons[0];
  const extension = createExtension({port, secret: first.secret, extensionId: first.extension_id, overrides});
  t.after(async () => {
    await extension.shutdown();
    for (const daemon of daemons) await daemon.stop().catch(() => {});
    fs.rmSync(workdir, {recursive: true, force: true});
  });
  return {
    port, workdir, extension,
    get daemon() { return daemons.at(-1); },
    // Same port and pairing, fresh process and state: an unclean daemon restart.
    restartDaemon: (options = {}) => launch(options),
  };
}

module.exports = {available, setup, sleep, waitFor, startDaemon, createExtension, freePort};
