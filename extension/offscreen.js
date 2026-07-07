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
    stopCapture();
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
  try {
    await beginSession(gen);
  } catch (err) {
    console.error("External session failed:", err);
    cleanup();
    chrome.runtime.sendMessage({ type: "CAPTURE_ERROR", message: err.message });
  }
}

function ingestExternalPcm(samples) {
  if (!capturing || !Array.isArray(samples) || samples.length === 0) return;
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
  sendStartMessage();

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
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(buffer);
  } else if (capturing) {
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

function stopCapture() {
  captureGen++;  // invalidate any in-flight connect/reconnect callbacks
  capturing = false;
  reconnecting = false;

  if (ws && ws.readyState === WebSocket.OPEN) {
    flushPcmBuffer();
    ws.send(JSON.stringify({ type: "stop" }));
  }

  // Small delay to let the stop message arrive before closing
  setTimeout(() => {
    cleanup();
    chrome.runtime.sendMessage({ type: "CAPTURE_STOPPED" });
  }, 500);
}

function cleanup() {
  capturing = false;
  reconnecting = false;
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
  const gen = captureGen; // do NOT bump — this must not invalidate a capture
  const connected = await connectWebSocket(1, gen);
  return connected && ws && ws.readyState === WebSocket.OPEN;
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
