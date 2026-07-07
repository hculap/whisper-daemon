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

// --- Message handling ---

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
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
    ingestExternalPcm(msg.samples);
    sendResponse({ ok: true });

  } else if (msg.type === "WD_GET_SETTINGS") {
    sendControl({ type: "get_settings" });
    sendResponse({ ok: true });

  } else if (msg.type === "WD_SET_SETTINGS") {
    sendControl({ type: "set_settings", patch: msg.patch || {} });
    sendResponse({ ok: true });

  } else if (msg.type === "WD_STOP_MEETING") {
    sendControl({ type: "stop_meeting" });
    sendResponse({ ok: true });
  }

  return true;
});

// --- Capture pipeline (legacy popup tab-capture) ---

async function startCaptureFromStream(streamId) {
  if (capturing) {
    console.warn("START_CAPTURE while already capturing — ignoring");
    chrome.runtime.sendMessage({ type: "CAPTURE_STARTED" });
    return;
  }
  const gen = ++captureGen;
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

    mediaStream.getAudioTracks().forEach((track) => {
      track.onended = () => {
        console.warn("Capture track ended");
        if (capturing) stopCapture();
      };
    });

    audioContext = new AudioContext({ sampleRate: 16000 });
    const source = audioContext.createMediaStreamSource(mediaStream);
    await audioContext.audioWorklet.addModule("pcm-processor.js");
    const workletNode = new AudioWorkletNode(audioContext, "pcm-processor");
    source.connect(workletNode);
    workletNode.connect(audioContext.destination);

    const ok = await beginSession(gen);
    if (!ok) return;

    workletNode.port.onmessage = (event) => {
      sendPcm(event.data.buffer);
    };
  } catch (err) {
    console.error("Capture failed:", err);
    cleanup();
    chrome.runtime.sendMessage({ type: "CAPTURE_ERROR", message: err.message });
  }
}

// --- External capture (content-script PCM) ---

async function startExternal() {
  if (capturing) {
    chrome.runtime.sendMessage({ type: "CAPTURE_STARTED" });
    return;
  }
  const gen = ++captureGen;
  externalMode = true;
  sessionStarting = true;
  try {
    await beginSession(gen);
  } catch (err) {
    console.error("External session failed:", err);
    cleanup();
    chrome.runtime.sendMessage({ type: "CAPTURE_ERROR", message: err.message });
  } finally {
    // Only clear if we still own the generation — a racing start/stop that
    // bumped captureGen owns the flag now.
    if (gen === captureGen) sessionStarting = false;
  }
}

function ingestExternalPcm(samples) {
  if (!Array.isArray(samples) || samples.length === 0) return;
  // Buffer even before the session is fully up so the first ~256-500ms of
  // audio is not lost while the socket opens and 'start' is sent.
  if (!capturing && !sessionStarting) return;
  const buffer = Float32Array.from(samples).buffer;
  sendPcm(buffer);
}

// --- Shared session bring-up (connect + start + ping) ---

async function beginSession(gen) {
  // Reuse an already-open socket (e.g. a control connection opened for
  // settings) instead of racing a second connection past the daemon's
  // single-connection slot.
  const alreadyOpen = ws && ws.readyState === WebSocket.OPEN;
  const connected = alreadyOpen || (await connectWebSocket(CONNECT_ATTEMPTS, gen));
  if (gen !== captureGen) return false; // stop/newer-start raced us
  if (!connected) {
    cleanup();
    chrome.runtime.sendMessage({
      type: "CAPTURE_ERROR",
      message: "Cannot connect to whisper-daemon. Is it running? Try: whisper-daemon restart",
    });
    return false;
  }

  capturing = true;
  sessionStarting = false;
  sendStartMessage();
  // Ship any audio buffered while the socket was coming up.
  flushPcmBuffer();

  clearInterval(pingInterval);
  pingInterval = setInterval(() => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "ping" }));
    }
  }, PING_INTERVAL_MS);

  chrome.runtime.sendMessage({ type: "CAPTURE_STARTED" });
  return true;
}

function sendStartMessage() {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({
    type: "start",
    meeting_title: meetTitle,
    meeting_url: meetUrl,
  }));
}

function sendPcm(buffer) {
  if (ws && ws.readyState === WebSocket.OPEN && capturing) {
    ws.send(buffer);
  } else if (capturing || sessionStarting) {
    bufferPcm(buffer);
  }
}

function bufferPcm(buffer) {
  if (pcmBufferedBytes + buffer.byteLength > MAX_BUFFERED_BYTES) {
    // Drop oldest audio first — recent speech matters more.
    while (pcmBuffer.length > 0 && pcmBufferedBytes + buffer.byteLength > MAX_BUFFERED_BYTES) {
      pcmBufferedBytes -= pcmBuffer[0].byteLength;
      pcmBuffer = pcmBuffer.slice(1);
    }
  }
  pcmBuffer = [...pcmBuffer, buffer];
  pcmBufferedBytes += buffer.byteLength;
}

function flushPcmBuffer() {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  const pending = pcmBuffer;
  pcmBuffer = [];
  pcmBufferedBytes = 0;
  for (const buffer of pending) {
    ws.send(buffer);
  }
}

// WD_STOP: end the meeting but KEEP the socket open. Stop forwarding PCM and
// tear down only the (legacy) capture pipeline; the daemon keeps pushing status
// over the live socket. Does NOT notify background (state stays CAPTURING until
// a real teardown) so the WD_STATUS relay tab pointer is preserved.
function stopForwarding() {
  const wasActive = capturing || sessionStarting;
  // Only invalidate in-flight connect/reconnect callbacks when we actually end
  // an active session. On a meet-left race the full-teardown STOP_CAPTURE runs
  // first (capturing already false); an unconditional bump here would advance
  // captureGen past the generation its delayed cleanup() timer captured, so the
  // timer would no-op and the socket + ping would leak (finding 1).
  if (wasActive) captureGen++;
  capturing = false;
  reconnecting = false;
  sessionStarting = false;
  externalMode = false;

  if (wasActive && ws && ws.readyState === WebSocket.OPEN) {
    flushPcmBuffer();
    ws.send(JSON.stringify({ type: "stop" }));
  } else if (wasActive) {
    // Ended an active session but the socket is not OPEN (mid-reconnect or
    // dropped): no 'stop' can be delivered and the daemon will never push a
    // terminal status. Surface an error so background relays WD_ERROR and the
    // in-bar button's optimistic 'transcribing' paint is reset (finding 2).
    chrome.runtime.sendMessage({
      type: "CAPTURE_ERROR",
      message: "Połączenie z whisper-daemon zostało utracone.",
    });
  }
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
}

// STOP_CAPTURE (meet-left / tab close / popup stop): full teardown — close the
// socket and everything else.
function stopCapture() {
  if (!capturing) {
    // Already stopped forwarding (WD_STOP keep-alive) — just close the socket.
    cleanup();
    chrome.runtime.sendMessage({ type: "CAPTURE_STOPPED" });
    return;
  }
  captureGen++;  // invalidate any in-flight connect/reconnect callbacks
  capturing = false;
  reconnecting = false;
  sessionStarting = false;

  if (ws && ws.readyState === WebSocket.OPEN) {
    flushPcmBuffer();
    ws.send(JSON.stringify({ type: "stop" }));
  }

  // Small delay to let the stop message arrive before closing. Generation-check
  // (F8): a start within 500ms of this stop must not be torn down by the stale
  // timer.
  const gen = captureGen;
  setTimeout(() => {
    if (gen !== captureGen) return;
    cleanup();
    chrome.runtime.sendMessage({ type: "CAPTURE_STOPPED" });
  }, 500);
}

function cleanup() {
  capturing = false;
  reconnecting = false;
  sessionStarting = false;
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
  const connected = await connectWebSocket(1, gen);
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

async function connectWebSocket(attempts, gen) {
  for (let attempt = 0; attempt < attempts; attempt++) {
    try {
      const socket = await openWebSocket(WS_URL);
      if (gen !== captureGen) {
        try { socket.close(); } catch {}
        return false;
      }
      ws = socket;
      ws.onmessage = handleServerMessage;
      ws.onclose = handleUnexpectedClose;
      return true;
    } catch {
      if (attempt < attempts - 1 && gen === captureGen) {
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

  if (data.type === "chunk_transcribed") {
    chrome.runtime.sendMessage({
      type: "CHUNK_TRANSCRIBED",
      text: data.text,
      startTime: data.start_time,
      chunkIndex: data.chunk_index,
    });
  } else if (data.type === "status") {
    chrome.runtime.sendMessage({
      type: "STATUS",
      state: data.state,
      elapsed: data.elapsed,
      chunks: data.chunks,
    });
  } else if (data.type === "settings") {
    chrome.runtime.sendMessage({
      type: "SETTINGS",
      settings: data.settings,
      devices: data.devices,
    });
  } else if (data.type === "error") {
    console.warn("Daemon error:", data.message);
    chrome.runtime.sendMessage({ type: "CAPTURE_ERROR", message: data.message });
  }
  // "pong" — ignore
}

function handleUnexpectedClose() {
  console.warn("WebSocket closed unexpectedly");
  ws = null;
  if (capturing && !reconnecting) {
    reconnectLoop(captureGen);
  }
}

async function reconnectLoop(gen) {
  reconnecting = true;
  chrome.runtime.sendMessage({ type: "CAPTURE_RECONNECTING" });
  const deadline = Date.now() + RECONNECT_WINDOW_MS;

  while (capturing && gen === captureGen && Date.now() < deadline) {
    const connected = await connectWebSocket(1, gen);
    if (connected && capturing && gen === captureGen) {
      reconnecting = false;
      console.info("Reconnected to whisper-daemon — resuming stream");
      sendStartMessage();
      flushPcmBuffer();
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

function openWebSocket(url) {
  return new Promise((resolve, reject) => {
    const socket = new WebSocket(url);
    socket.binaryType = "arraybuffer";
    socket.onopen = () => resolve(socket);
    socket.onerror = (err) => reject(err);
  });
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
