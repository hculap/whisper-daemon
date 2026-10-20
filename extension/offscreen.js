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
const START_ACK_TIMEOUT_MS = 10000;
// While the daemon saves the previous meeting it answers 'busy'; the new
// session keeps buffering and asks again (bounded by MAX_BUFFERED_BYTES).
const BUSY_RETRY_MS = 1000;
const BUSY_WAIT_MS = 5 * 60 * 1000;
const BUSY_MESSAGE = "Previous recording is still being saved, so the new one could not start.";

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

function captureStartedMessage() {
  return { type: "CAPTURE_STARTED", elapsed: pcmPosition / 16000 };
}

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
    chrome.runtime.sendMessage(captureStartedMessage());
    return;
  }
  const gen = prepareSession(false);
  try {
    if (!await deliverRetainedTail(gen)) return;
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        mandatory: {
          chromeMediaSource: "tab",
          chromeMediaSourceId: streamId,
        },
      },
      video: false,
    });

    if (gen !== captureGen) {
      mediaStream.getTracks().forEach(t => t.stop());
      settleSupersededStart();
      return;
    }
    mediaStream.getAudioTracks().forEach((track) => {
      track.onended = () => {
        console.warn("Capture track ended");
        if (capturing) stopCapture();
      };
    });

    audioContext = new AudioContext({ sampleRate: 16000 });
    const source = audioContext.createMediaStreamSource(mediaStream);
    await audioContext.audioWorklet.addModule("pcm-processor.js");
    if (gen !== captureGen) { settleSupersededStart(); return; }
    const workletNode = new AudioWorkletNode(audioContext, "pcm-processor");
    source.connect(workletNode);
    workletNode.connect(audioContext.destination);

    // Install before the handshake: preserve startup PCM and its sample clock.
    workletNode.port.onmessage = (event) => {
      sendPcm(event.data.samples.buffer, event.data.position);
    };
    if (!await beginSession(gen) && gen !== captureGen) settleSupersededStart();
  } catch (err) {
    failStart(gen, err);
  } finally {
    if (gen === captureGen) sessionStarting = false;
  }
}

// --- External capture (content-script PCM) ---

async function startExternal() {
  if (capturing || sessionStarting) {
    chrome.runtime.sendMessage(captureStartedMessage());
    return;
  }
  const gen = prepareSession(true);
  try {
    if (!await deliverRetainedTail(gen)) return;
    if (!await beginSession(gen) && gen !== captureGen) settleSupersededStart();
  } catch (err) {
    failStart(gen, err);
  } finally {
    // Only clear if we still own the generation — a racing start/stop that
    // bumped captureGen owns the flag now.
    if (gen === captureGen) sessionStarting = false;
  }
}

// New identity first: source PCM arriving while a previous failed tail is
// resolved, or while the daemon is busy, is staged under it, never discarded.
function prepareSession(external) {
  const gen = ++captureGen;
  sessionId = crypto.randomUUID();
  streamReady = startSent = false;
  pcmSequence = pcmPosition = 0;
  pcmBuffer = []; pcmBufferedBytes = 0;
  externalMode = external;
  sessionStarting = true;
  return gen;
}

// The previous session's undelivered stop comes first. When this start cannot
// continue, background has been told why (drain error or settled start).
async function deliverRetainedTail(gen) {
  if (stoppedSession) await drainStoppedSession();
  if (gen !== captureGen) { settleSupersededStart(); return false; }
  return !stoppedSession;
}

// A stop raced this start. Unless a newer session owns the state, tell
// background the start ended so it never stays in STARTING.
function settleSupersededStart() {
  if (!capturing && !sessionStarting) chrome.runtime.sendMessage({ type: "CAPTURE_STOPPED" });
}

function failStart(gen, err) {
  if (gen !== captureGen) { settleSupersededStart(); return; }
  console.error("Capture failed:", err);
  let message = err.message;
  // The daemon never owned this session; keep what was captured meanwhile.
  if (err.code === "busy" && pcmBuffer.length) {
    recoveryTails.push({id: sessionId, packets: pcmBuffer, position: pcmPosition});
    message += " Dźwięk nagrany w międzyczasie czeka na pobranie w popupie rozszerzenia.";
  }
  cleanup();
  chrome.runtime.sendMessage({ type: "CAPTURE_ERROR", message });
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
      message: connectionFailure === "offline"
        ? "Brak połączenia z whisper-daemon. Uruchom go: whisper-daemon restart"
        : "Sparuj rozszerzenie: otwórz jego popup i wklej klucz z Terminala.",
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

  chrome.runtime.sendMessage(captureStartedMessage());
  return true;
}

// Resolves on the daemon's 'started'. pendingStart() resolves it early (a stop
// raced the start); pendingStart.reject() fails it. A 'busy' reply means the
// daemon is saving another meeting and does not own this one yet: audio stays
// buffered and the start is re-sent on the next idle status or retry tick.
function sendStartMessage(id = sessionId, position = pcmPosition, track = true) {
  return new Promise((resolve, reject) => {
    const busyDeadline = Date.now() + BUSY_WAIT_MS;
    let timer = null;
    const entry = () => settle(resolve);
    const settle = (finish, value) => {
      clearTimeout(timer);
      if (pendingStart === entry) pendingStart = null;
      finish(value);
    };
    const send = () => {
      entry.waiting = false;
      if (!ws || ws.readyState !== WebSocket.OPEN) {
        settle(reject, new Error("Lost connection to whisper-daemon"));
        return;
      }
      clearTimeout(timer);
      timer = setTimeout(() => settle(reject, new Error("Daemon did not accept this recording")), START_ACK_TIMEOUT_MS);
      if (track) startSent = true;
      // Sample 0 was captured audio_elapsed ago, including any busy wait.
      const elapsed = (track ? pcmPosition : position) / 16000;
      ws.send(JSON.stringify({type: "start", session_id: id, meeting_title: meetTitle, meeting_url: meetUrl,
        audio_elapsed: Math.min(300, elapsed)}));
    };
    entry.sessionId = id;
    entry.waiting = false;
    entry.reject = error => settle(reject, error);
    entry.busy = () => {
      if (track) startSent = false;
      clearTimeout(timer);
      if (Date.now() >= busyDeadline) { settle(reject, busyError()); return; }
      entry.waiting = true;
      timer = setTimeout(send, BUSY_RETRY_MS);
    };
    entry.retry = () => { if (entry.waiting) send(); };
    pendingStart = entry;
    send();
  });
}

function busyError() {
  return Object.assign(new Error(BUSY_MESSAGE), {code: "busy"});
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

// Bounds only live, unacknowledged audio. Retained recovery tails are already
// bounded (each was once live) and must never block a new recording.
function bufferPcm(packet) {
  if (pcmBufferedBytes + packet.byteLength > MAX_BUFFERED_BYTES) {
    // Waited for the daemon as long as audio fits: give up on this start.
    if (pendingStart?.waiting && pendingStart.sessionId === sessionId) {
      pendingStart.reject(busyError());
      return false;
    }
    stopForwarding();
    chrome.runtime.sendMessage({type: "CAPTURE_ERROR", message: "Bufor połączenia audio jest pełny. Nagrywanie zatrzymane; dostarczony dźwięk jest zapisany."});
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
  // Cancel the old startup/reconnect generation before draining its snapshot,
  // but only when one is live. stopCapture() and starts compare captureGen
  // after awaiting a drain; a bump for an already-stopped session (a late
  // status, a second stop) would make them skip cleanup and their reply.
  if (capturing || sessionStarting || reconnecting) captureGen++;
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
    // A failed connect already reported CONNECTION; it is not a capture error.
    if (!await ensureControlConnection()) return;
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
async function openSharedSocket(attempts, gen) {
  const connectionGen = socketGen;
  for (let left = attempts; left > 0 && connectionGen === socketGen; left--) {
    if (ws && ws.readyState === WebSocket.OPEN) return true;
    if (!connectPromise) {
      connectPromise = connectWebSocket(left, gen).finally(() => {
        connectPromise = null;
      });
      return connectPromise;
    }
    // Joined another caller's attempt (often a single reconnect try). Its
    // failure costs this caller one attempt, not its whole budget.
    if (await connectPromise) return true;
    if (left > 1) await sleep(CONNECT_DELAY_MS);
  }
  return false;
}

async function connectWebSocket(attempts, gen) {
  const connectionGen = socketGen;
  let failure = "offline";
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
      reportConnection("connected");
      return true;
    } catch (err) {
      failure = err?.code || "offline";
      // A missing or rejected key does not fix itself between attempts.
      if (failure !== "offline") break;
      if (attempt < attempts - 1 && connectionGen === socketGen) {
        await sleep(CONNECT_DELAY_MS);
      }
    }
  }
  if (connectionGen === socketGen) {
    connectionFailure = failure;
    reportConnection(failure);
  }
  return false;
}

// Why the last connect failed, so a refused start names the right fix.
let connectionFailure = "offline";

// connected | offline | unpaired | rejected — drives the popup and Meet UI.
function reportConnection(state) {
  chrome.runtime.sendMessage({ type: "CONNECTION", state });
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
  } else if (data.type === "error" && data.code === "busy" && data.session_id === pendingStart?.sessionId) {
    pendingStart.busy();
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
    // Any idle means the previous meeting is saved: a waiting start may go.
    if (data.state === "idle") pendingStart?.retry();
    // Snapshots answer settings reads; only the owning session can terminate
    // capture. A delayed idle from another session is never authoritative.
    const live = capturing || sessionStarting;
    if (live && (data.snapshot || data.session_id !== sessionId)) return;
    if (live && (data.state === "transcribing" || data.state === "idle")) stopForwarding(false);
    // Otherwise the daemon's single meeting phase is relayed whatever session
    // it belongs to, so background never keeps a stale "saving" state.
    chrome.runtime.sendMessage({
      type: "STATUS",
      state: data.state,
      elapsed: data.elapsed,
      chunks: data.chunks,
      sessionId: data.session_id,
      snapshot: !!data.snapshot,
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
    // is not an error worth surfacing. No status can arrive now, so background
    // must not keep waiting for one (e.g. a "saving" state after a crash).
    console.debug("Control WebSocket closed — will reopen on demand");
    chrome.runtime.sendMessage({ type: "CONTROL_CLOSED" });
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
      chrome.runtime.sendMessage(captureStartedMessage());
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
      message: "Utracono połączenie z whisper-daemon i nie udało się go przywrócić.",
    });
  }
}

// Offscreen documents get chrome.runtime only (no chrome.storage), so the
// service worker reads the pairing key on our behalf.
async function readPairingSecret() {
  const reply = await chrome.runtime.sendMessage({ type: "GET_PAIRING_SECRET" });
  return reply?.secret || null;
}

function connectionError(message, code) {
  return Object.assign(new Error(message), { code });
}

async function openWebSocket(url) {
  const secret = await readPairingSecret();
  if (!secret) throw connectionError("Pair this extension in its popup first", "unpaired");
  return new Promise((resolve, reject) => {
    const socket = new WebSocket(url);
    let opened = false;
    const timer = setTimeout(() => { socket.close(); reject(connectionError("Authentication timeout", "offline")); }, 6000);
    socket.binaryType = "arraybuffer";
    socket.onopen = () => { opened = true; socket.send(JSON.stringify({type: "auth", secret})); };
    socket.onmessage = event => {
      let data;
      try { data = JSON.parse(event.data); } catch { return; }
      if (data.type === "authenticated") { clearTimeout(timer); resolve(socket); }
    };
    // Closed after opening but before authenticating: the daemon refused the key.
    socket.onclose = () => {
      clearTimeout(timer);
      reject(opened ? connectionError("Pairing rejected by daemon", "rejected")
                    : connectionError("Daemon unreachable", "offline"));
    };
    socket.onerror = () => { clearTimeout(timer); reject(connectionError("Daemon unreachable", "offline")); };
  });
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
