/**
 * Service worker: manages state machine, tabCapture, and offscreen document lifecycle.
 *
 * States: IDLE -> DETECTED -> STARTING -> CAPTURING
 *
 * The extension is a trigger + audio source for whisper-daemon.
 * When a meeting is detected, the user clicks "Start Capture" in the popup
 * (Chrome requires a user gesture for tabCapture). The daemon handles
 * recording, transcription, diarization, and output automatically.
 */

const State = { IDLE: "idle", DETECTED: "detected", STARTING: "starting", CAPTURING: "capturing" };

let state = State.IDLE;
let meetTabId = null;
let meetTitle = "";
let meetUrl = "";

// --- State machine ---

function transitionTo(newState) {
  state = newState;
  updateBadge();
}

function updateBadge() {
  const badges = {
    [State.IDLE]: { text: "", color: "#000" },
    [State.DETECTED]: { text: "!", color: "#FF9800" },
    [State.STARTING]: { text: "...", color: "#FF9800" },
    [State.CAPTURING]: { text: "REC", color: "#F44336" },
  };
  const badge = badges[state];
  chrome.action.setBadgeText({ text: badge.text });
  chrome.action.setBadgeBackgroundColor({ color: badge.color });
}

// --- Messages ---

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === "MEET_JOINED") {
    if (state === State.IDLE) {
      meetTabId = sender.tab?.id ?? null;
      meetTitle = msg.title || "";
      meetUrl = msg.url || "";
      transitionTo(State.DETECTED);

      chrome.notifications.create("meet-detected", {
        type: "basic",
        iconUrl: "icons/icon128.png",
        title: "Meeting Detected",
        message: `${meetTitle || "Google Meet"} — Click to start capture`,
        priority: 2,
        requireInteraction: true,
      });
    }
    sendResponse({ ok: true });

  } else if (msg.type === "MEET_LEFT") {
    if (state === State.CAPTURING || state === State.STARTING) {
      stopCapture();
    }
    meetTabId = null;
    transitionTo(State.IDLE);
    sendResponse({ ok: true });

  } else if (msg.type === "CAPTURE_STARTED") {
    transitionTo(State.CAPTURING);
    relayToContentScript(msg);
    sendResponse({ ok: true });

  } else if (msg.type === "CAPTURE_STOPPED") {
    relayToContentScript(msg);
    transitionTo(State.IDLE);
    meetTabId = null;
    sendResponse({ ok: true });

  } else if (msg.type === "CAPTURE_ERROR") {
    chrome.notifications.create("capture-error", {
      type: "basic",
      iconUrl: "icons/icon128.png",
      title: "Capture Error",
      message: msg.message || "Failed to capture audio",
    });
    relayToContentScript(msg);
    transitionTo(State.IDLE);
    sendResponse({ ok: true });

  } else if (msg.type === "CHUNK_TRANSCRIBED") {
    relayToContentScript(msg);
    sendResponse({ ok: true });

  } else if (msg.type === "GET_STATE") {
    sendResponse({ state, meetTabId, meetTitle, meetUrl });

  } else if (msg.type === "START_WITH_STREAM") {
    // Popup obtained streamId (has user gesture for tabCapture)
    if (state !== State.CAPTURING) {
      meetTabId = msg.tabId || meetTabId;
      transitionTo(State.STARTING);
      startCaptureWithStream(msg.streamId);
    }
    sendResponse({ ok: true });

  } else if (msg.type === "STOP_RECORDING") {
    if (state === State.CAPTURING || state === State.STARTING) {
      stopCapture();
    }
    sendResponse({ ok: true });
  }

  return true;
});

// --- Notification click: open popup ---

chrome.notifications.onClicked.addListener((notificationId) => {
  if (notificationId === "meet-detected" && state === State.DETECTED && meetTabId) {
    chrome.notifications.clear("meet-detected");
    chrome.tabs.update(meetTabId, { active: true });
    chrome.action.openPopup();
  }
});

// --- Capture ---

async function startCaptureWithStream(streamId) {
  try {
    await ensureOffscreenDocument();

    chrome.runtime.sendMessage({
      type: "START_CAPTURE",
      streamId,
      tabId: meetTabId,
      meetTitle,
      meetUrl,
    });
  } catch (err) {
    console.error("Failed to start capture:", err);
    chrome.runtime.sendMessage({
      type: "CAPTURE_ERROR",
      message: err.message,
    });
  }
}

function stopCapture() {
  chrome.runtime.sendMessage({ type: "STOP_CAPTURE" });
}

async function ensureOffscreenDocument() {
  const contexts = await chrome.runtime.getContexts({
    contextTypes: ["OFFSCREEN_DOCUMENT"],
  });
  if (contexts.length > 0) return;

  await chrome.offscreen.createDocument({
    url: "offscreen.html",
    reasons: ["USER_MEDIA"],
    justification: "Capture tab audio for transcription",
  });
}

// --- Relay to content script (overlay) ---

function relayToContentScript(msg) {
  if (meetTabId) {
    chrome.tabs.sendMessage(meetTabId, msg).catch(() => {});
  }
}

// --- Cleanup on tab close ---

chrome.tabs.onRemoved.addListener((tabId) => {
  if (tabId === meetTabId) {
    if (state === State.CAPTURING) {
      stopCapture();
    }
    meetTabId = null;
    transitionTo(State.IDLE);
  }
});
