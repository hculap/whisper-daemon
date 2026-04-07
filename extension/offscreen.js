/**
 * Offscreen document — captures tab audio, extracts PCM via AudioWorklet,
 * and streams to whisper-daemon over WebSocket.
 */

const WS_URL = "ws://127.0.0.1:9876/ws/audio";
const RECONNECT_ATTEMPTS = 3;
const RECONNECT_DELAY_MS = 1000;
const PING_INTERVAL_MS = 25000;

let audioContext = null;
let mediaStream = null;
let ws = null;
let pingInterval = null;
let meetTitle = "";
let meetUrl = "";

// --- Message handling ---

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg.type === "START_CAPTURE") {
    meetTitle = msg.meetTitle || "";
    meetUrl = msg.meetUrl || "";
    startCapture(msg.streamId);
    sendResponse({ ok: true });

  } else if (msg.type === "STOP_CAPTURE") {
    stopCapture();
    sendResponse({ ok: true });
  }

  return true;
});

// --- Capture pipeline ---

async function startCapture(streamId) {
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

    // 16kHz to match whisper-daemon's expected sample rate
    audioContext = new AudioContext({ sampleRate: 16000 });
    const source = audioContext.createMediaStreamSource(mediaStream);

    await audioContext.audioWorklet.addModule("pcm-processor.js");
    const workletNode = new AudioWorkletNode(audioContext, "pcm-processor");

    // Connect source -> worklet -> destination (destination plays audio back)
    source.connect(workletNode);
    workletNode.connect(audioContext.destination);

    // Connect WebSocket
    const connected = await connectWebSocket();
    if (!connected) {
      cleanup();
      chrome.runtime.sendMessage({
        type: "CAPTURE_ERROR",
        message: "Cannot connect to whisper-daemon. Run: whisper-daemon serve",
      });
      return;
    }

    // Send start message
    ws.send(JSON.stringify({
      type: "start",
      meeting_title: meetTitle,
      meeting_url: meetUrl,
    }));

    // Stream PCM chunks
    workletNode.port.onmessage = (event) => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(event.data.buffer);
      }
    };

    // Keepalive
    pingInterval = setInterval(() => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "ping" }));
      }
    }, PING_INTERVAL_MS);

    chrome.runtime.sendMessage({ type: "CAPTURE_STARTED" });

  } catch (err) {
    console.error("Capture failed:", err);
    cleanup();
    chrome.runtime.sendMessage({
      type: "CAPTURE_ERROR",
      message: err.message,
    });
  }
}

function stopCapture() {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "stop" }));
  }

  // Small delay to let the stop message arrive before closing
  setTimeout(() => {
    cleanup();
    chrome.runtime.sendMessage({ type: "CAPTURE_STOPPED" });
  }, 500);
}

function cleanup() {
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

// --- WebSocket ---

async function connectWebSocket() {
  for (let attempt = 0; attempt < RECONNECT_ATTEMPTS; attempt++) {
    try {
      ws = await openWebSocket(WS_URL);

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === "chunk_transcribed") {
            // Forward live transcription to popup via service worker
            chrome.runtime.sendMessage({
              type: "CHUNK_TRANSCRIBED",
              text: data.text,
              startTime: data.start_time,
              chunkIndex: data.chunk_index,
            });
          }
        } catch {
          // binary or unparseable — ignore
        }
      };

      ws.onclose = () => {
        console.warn("WebSocket closed unexpectedly");
        cleanup();
        chrome.runtime.sendMessage({ type: "CAPTURE_STOPPED" });
      };

      return true;
    } catch {
      if (attempt < RECONNECT_ATTEMPTS - 1) {
        await sleep(RECONNECT_DELAY_MS);
      }
    }
  }
  return false;
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
