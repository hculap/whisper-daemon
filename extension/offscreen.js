/**
 * Offscreen document — owns the WebSocket to whisper-daemon (extension CSP
 * requires ws:// to originate here, not from the content script).
 *
 * Audio no longer originates here for the Meet flow: PCM arrives as WD_PCM
 * messages relayed from the content script's AudioWorklet. The legacy popup
 * fallback still captures via getUserMedia(streamId) below.
 *
 * Resilience is unchanged: if the socket drops mid-capture, PCM is buffered
 * (bounded) and a reconnect loop runs for up to RECONNECT_WINDOW_MS.
 *
 * The socket is a PERSISTENT per-session control channel: WD_STOP ends the
 * *meeting* (stops forwarding PCM) but KEEPS the socket open so the daemon's
 * terminal transcribing→idle status still lands and resets the in-bar button.
 * The socket is only closed on a full teardown (STOP_CAPTURE — meet-left / tab
 * close) or when the reconnect window is exhausted.
 */

const WS_URL = "ws://127.0.0.1:9876";
const CONNECT_ATTEMPTS = 10;
const CONNECT_DELAY_MS = 1000;
const RECONNECT_INTERVAL_MS = 2000;
const RECONNECT_WINDOW_MS = 5 * 60 * 1000;
const PING_INTERVAL_MS = 25000;
// 16kHz float32 mono = 64 KB/s; 8 MB ≈ 2 minutes of buffered audio.
const MAX_BUFFERED_BYTES = 8 * 1024 * 1024;

let audioContext = null;
let mediaStream = null;
let ws = null;
let pingInterval = null;
let meetTitle = "";
let meetUrl = "";
let capturing = false;
let reconnecting = false;
// True from the moment a session begins bringing up (WD_START) until it is
// fully capturing. Lets us buffer the first PCM frames instead of dropping
// them while the socket opens and the 'start' message is sent.
let sessionStarting = false;
let externalMode = false; // true = PCM arrives via WD_PCM (content script)
let pcmBuffer = [];
let pcmBufferedBytes = 0;
// Bumped on every start AND stop. Async callbacks (connect attempts,
// reconnect loop, delayed cleanup) capture the value at scheduling time
// and bail if it no longer matches — so a socket that finishes opening
// after a stop, or a stale reconnect loop, can't leak into a new session.
let captureGen = 0;
let socketGen = 0; // control connections survive a new capture generation
let sessionId = "";
let pcmSequence = 0;
let pcmPosition = 0;
let pendingStart = null;
let streamReady = false; // start ACK + ordered replay completed on this socket
let startSent = false; // daemon may own a session even before its ACK arrives
let stopPromise = null;
let stoppedSession = null; // retain undelivered PCM until Stop can reach daemon
const recoveryTails = [];

// --- Message handling ---

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  // Content-script broadcasts also reach this document. Only consume the
  // explicitly routed copy, never both the original and the SW relay.
  if (msg.target !== "offscreen") return false;
  if (msg.type === 'GET_RECOVERY') {
    sendResponse({count:recoveryTails.length, pending:!!stoppedSession});return;
  }
  if (msg.type === 'EXPORT_RECOVERY') {
    const tail=recoveryTails[0] || stoppedSession;
    if (!tail || !tail.packets.length) {sendResponse({error:'Brak bufora audio'});return;}
    const {blob,offset}=recoveryWav(tail.packets);
    if(tail.url)URL.revokeObjectURL(tail.url);
    tail.url=URL.createObjectURL(blob);
    sendResponse({url:tail.url,filename:`whisper-tail-${tail.id}-from-${Math.round(offset*1000)}ms.wav`});return;
  }
  if (msg.type === 'CLEAR_EXPORTED_RECOVERY') {
    // Explicit user action after checking the downloaded file; never automatic.
    const tail=recoveryTails[0] || stoppedSession;
    if(tail?.url){URL.revokeObjectURL(tail.url);if(recoveryTails[0]===tail)recoveryTails.shift();else stoppedSession=null;}
    sendResponse({ok:true});return;
  }
  if (msg.type === "START_CAPTURE") {
    // Legacy popup fallback: capture the tab here via getUserMedia.
    meetTitle = msg.meetTitle || "";
    meetUrl = msg.meetUrl || "";
    startCaptureFromStream(msg.streamId);
    sendResponse({ ok: true });

  } else if (msg.type === "STOP_CAPTURE") {
    stopCapture();
    sendResponse({ ok: true });

  } else if (msg.type === "WD_START") {
    // New Meet flow: audio arrives externally as WD_PCM.
    meetTitle = msg.title || "";
    meetUrl = msg.url || "";
    startExternal();
    sendResponse({ ok: true });

  } else if (msg.type === "WD_STOP") {
    // Meeting ended — stop forwarding PCM but KEEP the socket open so the
    // daemon's terminal transcribing→idle status still reaches the content
    // script and resets the button.
    stopForwarding();
    sendResponse({ ok: true });

  } else if (msg.type === "WD_PCM") {
    ingestExternalPcm(msg.samples, msg.position);
    sendResponse({ ok: true });

  } else if (msg.type === "WD_GET_SETTINGS") {
    sendControl({ type: "get_settings" });
    sendResponse({ ok: true });

  } else if (msg.type === "WD_SET_SETTINGS") {
    sendControl({ type: "set_settings", patch: msg.patch || {} });
    sendResponse({ ok: true });

  } else if (msg.type === "WD_PAUSE_MEETING") {
    sendControl({ type: "pause_meeting" });
    sendResponse({ ok: true });

  } else if (msg.type === "WD_STOP_MEETING") {
    sendControl({ type: "stop_meeting" });
    sendResponse({ ok: true });
  }

  return true;
});

// --- Capture pipeline (legacy popup tab-capture) ---

async function startCaptureFromStream(streamId) {
  if (capturing || sessionStarting) {
    console.warn("START_CAPTURE while already capturing — ignoring");
    chrome.runtime.sendMessage({ type: "CAPTURE_STARTED" });
    return;
  }
  const gen = ++captureGen;
  if (stoppedSession) await drainStoppedSession();
  if (gen !== captureGen || stoppedSession) return;
  sessionId = crypto.randomUUID();
  streamReady = startSent = false;
  pcmSequence = pcmPosition = 0;
  sessionStarting = true;
  externalMode = false;
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        mandatory: {
          chromeMediaSource: "tab",
          chromeMediaSourceId: streamId,
        },
      },
      video: false,
    });

    if (gen !== captureGen) { mediaStream.getTracks().forEach(t => t.stop()); return; }
    mediaStream.getAudioTracks().forEach((track) => {
      track.onended = () => {
        console.warn("Capture track ended");
        if (capturing) stopCapture();
      };
    });

    audioContext = new AudioContext({ sampleRate: 16000 });
    const source = audioContext.createMediaStreamSource(mediaStream);
    await audioContext.audioWorklet.addModule("pcm-processor.js");
    if (gen !== captureGen) return;
    const workletNode = new AudioWorkletNode(audioContext, "pcm-processor");
    source.connect(workletNode);
    workletNode.connect(audioContext.destination);

    // Install before the handshake: preserve startup PCM and its sample clock.
    workletNode.port.onmessage = (event) => {
      sendPcm(event.data.samples.buffer, event.data.position);
    };
    const ok = await beginSession(gen);
    if (!ok) return;
  } catch (err) {
    console.error("Capture failed:", err);
    cleanup();
    chrome.runtime.sendMessage({ type: "CAPTURE_ERROR", message: err.message });
  }
}

// --- External capture (content-script PCM) ---

async function startExternal() {
  if (capturing || sessionStarting) {
    chrome.runtime.sendMessage({ type: "CAPTURE_STARTED" });
    return;
  }
  const gen = ++captureGen;
  sessionId = crypto.randomUUID();
  streamReady = startSent = false;
  pcmSequence = pcmPosition = 0;
  pcmBuffer = []; pcmBufferedBytes = 0;
  externalMode = true;
  sessionStarting = true;
  try {
    // New source PCM may already be arriving while the previous failed tail
    // is being resolved. Stage it under the new identity, never discard it.
    if (stoppedSession) await drainStoppedSession();
    if (gen !== captureGen || stoppedSession) return;
    await beginSession(gen);
  } catch (err) {
    if (gen !== captureGen) return;
    console.error("External session failed:", err);
    cleanup();
    chrome.runtime.sendMessage({ type: "CAPTURE_ERROR", message: err.message });
  } finally {
    // Only clear if we still own the generation — a racing start/stop that
    // bumped captureGen owns the flag now.
    if (gen === captureGen) sessionStarting = false;
  }
}

function ingestExternalPcm(samples, position) {
  if (!Array.isArray(samples) || samples.length === 0 || samples.length > 16384 ||
      samples.some(v => !Number.isFinite(v) || Math.abs(v) > 1.01)) return;
  // Buffer even before the session is fully up so the first ~256-500ms of
  // audio is not lost while the socket opens and 'start' is sent.
  if (!capturing && !sessionStarting) return;
  const buffer = Float32Array.from(samples).buffer;
  sendPcm(buffer, position);
}

// --- Shared session bring-up (connect + start + ping) ---

async function beginSession(gen) {
  // Reuse an already-open socket (e.g. a control connection opened for
  // settings) instead of racing a second connection past the daemon's
  // single-connection slot.
  const alreadyOpen = ws && ws.readyState === WebSocket.OPEN;
  const connected = alreadyOpen || (await openSharedSocket(CONNECT_ATTEMPTS, gen));
  if (gen !== captureGen) return false; // stop/newer-start raced us
  if (!connected) {
    cleanup();
    chrome.runtime.sendMessage({
      type: "CAPTURE_ERROR",
      message: "Cannot connect to whisper-daemon. Is it running? Try: whisper-daemon restart",
    });
    return false;
  }

  await sendStartMessage();
  if (gen !== captureGen) return false;
  capturing = true;
  sessionStarting = false;
  // Ship any audio buffered while the socket was coming up.
  flushPcmBuffer();
  streamReady = true;

  clearInterval(pingInterval);
  pingInterval = setInterval(() => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "ping" }));
    }
  }, PING_INTERVAL_MS);

  chrome.runtime.sendMessage({ type: "CAPTURE_STARTED" });
  return true;
}

function sendStartMessage(id = sessionId, position = pcmPosition, track = true) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => { pendingStart = null; reject(new Error("Daemon did not accept this recording")); }, 10000);
    pendingStart = () => { clearTimeout(timer); pendingStart = null; resolve(); };
    pendingStart.sessionId = id;
    pendingStart.reject = error => { clearTimeout(timer); pendingStart = null; reject(error); };
    if (track) startSent = true;
    ws.send(JSON.stringify({type: "start", session_id: id, meeting_title: meetTitle, meeting_url: meetUrl,
      audio_elapsed: Math.min(300, position / 16000)}));
  });
}

function sendPcm(buffer, position) {
  if (!capturing && !sessionStarting) return;
  if (position !== undefined && (!Number.isSafeInteger(position) || position < pcmPosition)) return;
  const start = position === undefined ? pcmPosition : position;
  const packet = new ArrayBuffer(16 + buffer.byteLength);
  const header = new DataView(packet);
  header.setUint32(0, 0x31414457, true); // WDA1
  header.setUint32(4, pcmSequence++, true);
  header.setBigUint64(8, BigInt(start), true);
  new Uint8Array(packet, 16).set(new Uint8Array(buffer));
  pcmPosition = start + buffer.byteLength / 4;
  if (!bufferPcm(packet)) return;
  if (ws && ws.readyState === WebSocket.OPEN && streamReady) ws.send(packet);
}

function bufferPcm(packet) {
  const recoveryBytes=recoveryTails.reduce((total,tail)=>total+tail.packets.reduce((sum,p)=>sum+p.byteLength,0),0);
  if (pcmBufferedBytes + packet.byteLength + recoveryBytes > MAX_BUFFERED_BYTES) {
    stopForwarding();
    chrome.runtime.sendMessage({type: "CAPTURE_ERROR", message: "Audio connection buffer is full. Recording stopped; already delivered audio is saved."});
    return false;
  }
  pcmBuffer.push(packet);
  pcmBufferedBytes += packet.byteLength;
  return true;
}

function flushPcmBuffer() {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  for (const packet of pcmBuffer) ws.send(packet);
}

// WD_STOP: end the meeting but KEEP the socket open. Stop forwarding PCM and
// tear down only the (legacy) capture pipeline; the daemon keeps pushing status
// over the live socket. Does NOT notify background (state stays CAPTURING until
// a real teardown) so the WD_STATUS relay tab pointer is preserved.
function stopForwarding(notifyDaemon = true) {
  const canFlush = streamReady;
  streamReady = false;
  // Cancel the old startup/reconnect generation before draining its snapshot.
  captureGen++;
  capturing = false;
  reconnecting = false;
  sessionStarting = false;
  externalMode = false;

  let stopped = Promise.resolve(true);
  if (notifyDaemon && startSent && ws && ws.readyState === WebSocket.OPEN) {
    if (canFlush) {
      flushPcmBuffer();
      ws.send(JSON.stringify({ type: "stop", session_id: sessionId }));
    } else {
      stoppedSession = {id:sessionId, packets:pcmBuffer, position:pcmPosition};
    }
  } else if (notifyDaemon && startSent) {
    // An authenticated control connection can stop a disconnected session
    // without starting capture again. Keep teardown alive until this is sent.
    stoppedSession = {id:sessionId, packets:pcmBuffer, position:pcmPosition};
  }
  startSent = false;
  if (pendingStart?.sessionId === sessionId && !stopPromise) pendingStart();
  if (notifyDaemon && stoppedSession) stopped = drainStoppedSession();
  // A new session must not inherit audio that never made it out.
  pcmBuffer = [];
  pcmBufferedBytes = 0;

  // Tear down ONLY the capture pipeline (legacy tab-capture path). Keep ws +
  // ping alive so status/settings continue to flow.
  if (audioContext) {
    audioContext.close().catch(() => {});
    audioContext = null;
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }
  return stopped;
}

function drainStoppedSession() {
  if (stopPromise) return stopPromise;
  const ending = stoppedSession;
  stopPromise = (async () => {
    try {
      const connected = await openSharedSocket(CONNECT_ATTEMPTS, captureGen);
      if (!connected) throw new Error('Daemon unavailable');
      await sendStartMessage(ending.id, ending.position, false);
      for (const packet of ending.packets) ws.send(packet);
      ws.send(JSON.stringify({type:'stop',session_id:ending.id}));
      if (stoppedSession === ending) stoppedSession = null;
      return true;
    } catch (error) {
      if(stoppedSession===ending)chrome.runtime.sendMessage({type:'CAPTURE_ERROR',message:'Nie udało się przekazać końcówki audio. Pozostaje w buforze rozszerzenia; po przywróceniu połączenia ponów Stop albo pobierz WAV z popupu. Mikrofon można zatrzymać w menu aplikacji.'});
      return false;
    } finally { stopPromise = null; }
  })();
  return stopPromise;
}

// STOP_CAPTURE (meet-left / tab close / popup stop): full teardown — close the
// socket and everything else.
async function stopCapture() {
  const stopped = stopForwarding();
  const gen = captureGen;
  if (!await stopped) return; // do not discard the retained tail
  if (gen !== captureGen) return;
  cleanup();
  chrome.runtime.sendMessage({ type: "CAPTURE_STOPPED" });
}

function cleanup() {
  captureGen++; // also invalidate pending opens, before ws has been assigned
  socketGen++;
  capturing = false;
  reconnecting = false;
  sessionStarting = false;
  streamReady = startSent = false;
  if (pendingStart) pendingStart();
  externalMode = false;
  pcmBuffer = [];
  pcmBufferedBytes = 0;

  clearInterval(pingInterval);
  pingInterval = null;

  if (ws) {
    ws.onclose = null; // prevent reconnect on intentional close
    ws.close();
    ws = null;
  }

  if (audioContext) {
    audioContext.close().catch(() => {});
    audioContext = null;
  }

  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }
}

// --- Control-only connection (settings / status without recording) ---

async function ensureControlConnection() {
  if (ws && ws.readyState === WebSocket.OPEN) return true;
  // A capture reconnect loop already owns the single socket (ws is transiently
  // null between attempts). Opening a competing control socket would race past
  // the daemon's single-connection slot — wait for the reconnecting socket
  // instead (F7-js/F12-js).
  if (reconnecting) return await waitForReconnect();
  const gen = captureGen; // do NOT bump — this must not invalidate a capture
  const connected = await openSharedSocket(1, gen);
  return connected && ws && ws.readyState === WebSocket.OPEN;
}

// Wait (bounded) for an in-progress capture reconnect to re-establish the
// single socket, so queued control messages ride the reconnecting connection.
async function waitForReconnect(timeoutMs = RECONNECT_INTERVAL_MS * 3) {
  const deadline = Date.now() + timeoutMs;
  while (reconnecting && Date.now() < deadline) {
    if (ws && ws.readyState === WebSocket.OPEN) return true;
    await sleep(100);
  }
  return !!(ws && ws.readyState === WebSocket.OPEN);
}

async function sendControl(obj) {
  try {
    const ok = await ensureControlConnection();
    if (!ok) {
      chrome.runtime.sendMessage({
        type: "CAPTURE_ERROR",
        message: "whisper-daemon unreachable for settings.",
      });
      return;
    }
    ws.send(JSON.stringify(obj));
  } catch (err) {
    console.warn("Control message failed:", err);
  }
}

// --- WebSocket ---

// Serialize socket bring-up: concurrent callers (two WD_GET_SETTINGS on boot,
// or a settings fetch racing a record start) must share ONE connection attempt.
// Without this each opens its own socket, the daemon keeps only the newest
// ("New browser connection — closing previous one") and the losing socket fires
// a spurious "WebSocket closed unexpectedly" on the client.
let connectPromise = null;
function openSharedSocket(attempts, gen) {
  if (ws && ws.readyState === WebSocket.OPEN) return Promise.resolve(true);
  if (connectPromise) return connectPromise;
  connectPromise = connectWebSocket(attempts, gen).finally(() => {
    connectPromise = null;
  });
  return connectPromise;
}

async function connectWebSocket(attempts, gen) {
  const connectionGen = socketGen;
  for (let attempt = 0; attempt < attempts; attempt++) {
    try {
      const socket = await openWebSocket(WS_URL);
      if (connectionGen !== socketGen) {
        try { socket.close(); } catch {}
        return false;
      }
      ws = socket;
      ws.onmessage = handleServerMessage;
      ws.onclose = handleUnexpectedClose;
      return true;
    } catch {
      if (attempt < attempts - 1 && connectionGen === socketGen) {
        await sleep(CONNECT_DELAY_MS);
      }
    }
  }
  return false;
}

function handleServerMessage(event) {
  let data;
  try {
    data = JSON.parse(event.data);
  } catch {
    return; // binary or unparseable — ignore
  }

  if (data.type === "started" && data.session_id === pendingStart?.sessionId) {
    pendingStart();
  } else if (data.type === "audio_ack" && data.session_id === sessionId) {
    while (pcmBuffer.length && new DataView(pcmBuffer[0]).getUint32(4, true) <= data.sequence) {
      pcmBufferedBytes -= pcmBuffer.shift().byteLength;
    }
  } else if (data.type === "health") {
    chrome.runtime.sendMessage({type: "HEALTH", sources: data.sources, paused: data.paused});
  } else if (data.type === "capabilities") {
    chrome.runtime.sendMessage({type: "CAPABILITIES", diarization: data.diarization});
  } else if (data.type === "chunk_transcribed") {
    chrome.runtime.sendMessage({
      type: "CHUNK_TRANSCRIBED",
      text: data.text,
      startTime: data.start_time,
      chunkIndex: data.chunk_index,
    });
  } else if (data.type === "status") {
    // Snapshots answer settings reads; only the owning session can terminate
    // capture. A delayed idle from another session is never authoritative.
    if ((capturing || sessionStarting) && (data.snapshot || data.session_id !== sessionId)) return;
    if (!data.snapshot && data.session_id && data.session_id !== sessionId) return;
    if (data.state === "transcribing" || data.state === "idle") {
      stopForwarding(false);
    }
    chrome.runtime.sendMessage({
      type: "STATUS",
      state: data.state,
      elapsed: data.elapsed,
      chunks: data.chunks,
      sessionId: data.session_id,
    });
  } else if (data.type === "settings") {
    chrome.runtime.sendMessage({
      type: "SETTINGS",
      settings: data.settings,
      devices: data.devices,
    });
  } else if (data.type === "error") {
    if(data.code==='session_ended' && stoppedSession?.id===data.session_id){
      recoveryTails.push(stoppedSession);stoppedSession=null;
      // The rejected socket is closing. A new session must use a fresh one.
      if(ws){ws.onclose=null;ws.close();ws=null;}
      if(pendingStart)pendingStart.reject(new Error('Session ended; tail retained for local export'));
      chrome.runtime.sendMessage({type:'RECOVERY_AVAILABLE',message:'Poprzednie spotkanie jest zamknięte. Końcówka audio czeka na lokalny eksport w popupie rozszerzenia. Możesz rozpocząć nowe nagranie.'});
      return;
    }
    console.warn("Daemon error:", data.message);
    chrome.runtime.sendMessage({ type: "CAPTURE_ERROR", message: data.message });
  }
  // "pong" — ignore
}

function handleUnexpectedClose() {
  ws = null;
  streamReady = false;
  if (capturing && !reconnecting) {
    console.warn("WebSocket closed during recording — reconnecting");
    reconnectLoop(captureGen);
  } else {
    // Benign: a daemon restart, or a superseded/idle control connection. The
    // channel reopens on demand (next settings fetch or record start), so this
    // is not an error worth surfacing.
    console.debug("Control WebSocket closed — will reopen on demand");
  }
}

async function reconnectLoop(gen) {
  reconnecting = true;
  chrome.runtime.sendMessage({ type: "CAPTURE_RECONNECTING" });
  const deadline = Date.now() + RECONNECT_WINDOW_MS;

  while (capturing && gen === captureGen && Date.now() < deadline) {
    const connected = await openSharedSocket(1, gen);
    if (connected && capturing && gen === captureGen) {
      console.info("Reconnected to whisper-daemon — resuming stream");
      try { await sendStartMessage(); } catch { cleanup(); return; }
      if (gen !== captureGen) return;
      flushPcmBuffer();
      streamReady = true;
      reconnecting = false;
      chrome.runtime.sendMessage({ type: "CAPTURE_STARTED" });
      return;
    }
    await sleep(RECONNECT_INTERVAL_MS);
  }

  reconnecting = false;
  if (capturing && gen === captureGen) {
    console.error("Could not reconnect to whisper-daemon — stopping capture");
    cleanup();
    chrome.runtime.sendMessage({
      type: "CAPTURE_ERROR",
      message: "Lost connection to whisper-daemon and could not reconnect.",
    });
  }
}

async function openWebSocket(url) {
  const pairing = await chrome.storage.local.get("pairingSecret");
  if (!pairing.pairingSecret) throw new Error("Pair this extension in its popup first");
  return new Promise((resolve, reject) => {
    const socket = new WebSocket(url);
    const timer = setTimeout(() => { socket.close(); reject(new Error("Authentication timeout")); }, 6000);
    socket.binaryType = "arraybuffer";
    socket.onopen = () => socket.send(JSON.stringify({type: "auth", secret: pairing.pairingSecret}));
    socket.onmessage = event => {
      let data;
      try { data = JSON.parse(event.data); } catch { return; }
      if (data.type === "authenticated") { clearTimeout(timer); resolve(socket); }
    };
    socket.onclose = () => { clearTimeout(timer); reject(new Error("Pairing rejected by daemon")); };
    socket.onerror = err => { clearTimeout(timer); reject(err); };
  });
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
