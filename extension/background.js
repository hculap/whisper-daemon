/**
 * Service worker: manages state machine, tabCapture, and offscreen document lifecycle.
 *
 * States: IDLE -> DETECTED -> CAPTURING
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

// --- Messages from content script and offscreen ---

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
        message: `${meetTitle || "Google Meet"} — Click to start recording`,
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
    sendResponse({ ok: true });

  } else if (msg.type === "CAPTURE_STOPPED") {
    transitionTo(State.IDLE);
    meetTabId = null;
    sendResponse({ ok: true });

  } else if (msg.type === "CAPTURE_ERROR") {
    chrome.notifications.create("capture-error", {
      type: "basic",
      iconUrl: "icons/icon128.png",
      title: "Recording Error",
      message: msg.message || "Failed to capture audio",
    });
    transitionTo(State.IDLE);
    sendResponse({ ok: true });

  } else if (msg.type === "GET_STATE") {
    sendResponse({ state, meetTabId, meetTitle, meetUrl });

  } else if (msg.type === "START_RECORDING") {
    if (state === State.DETECTED && meetTabId) {
      transitionTo(State.STARTING);
      startCapture(meetTabId);
    } else if (state === State.IDLE) {
      transitionTo(State.STARTING);
      chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        if (tabs[0]) {
          meetTabId = tabs[0].id;
          meetTitle = tabs[0].title || "";
          meetUrl = tabs[0].url || "";
          startCapture(meetTabId);
        } else {
          transitionTo(State.IDLE);
        }
      });
    }
    // STARTING or CAPTURING — ignore duplicate clicks
    sendResponse({ ok: true });

  } else if (msg.type === "STOP_RECORDING") {
    if (state === State.CAPTURING || state === State.STARTING) {
      stopCapture();
    }
    sendResponse({ ok: true });
  }

  return true; // async response
});

// --- Notification click ---

chrome.notifications.onClicked.addListener((notificationId) => {
  if (notificationId === "meet-detected" && state === State.DETECTED && meetTabId) {
    chrome.notifications.clear("meet-detected");
    transitionTo(State.STARTING);
    startCapture(meetTabId);
  }
});

// --- Tab capture + offscreen orchestration ---

async function startCapture(tabId) {
  try {
    const streamId = await new Promise((resolve, reject) => {
      chrome.tabCapture.getMediaStreamId({ targetTabId: tabId }, (id) => {
        if (chrome.runtime.lastError) {
          return reject(new Error(chrome.runtime.lastError.message));
        }
        if (!id) return reject(new Error("Empty stream ID"));
        resolve(id);
      });
    });

    await ensureOffscreenDocument();

    chrome.runtime.sendMessage({
      type: "START_CAPTURE",
      streamId,
      tabId,
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
