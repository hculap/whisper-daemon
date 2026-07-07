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
    // Let the in-page orchestrator re-init its control/observer/hosts after an
    // SPA re-join (content scripts are not reloaded on Meet route changes).
    if (sender.tab?.id) {
      chrome.tabs.sendMessage(sender.tab.id, { type: "WD_MEET_JOINED" }).catch(() => {});
    }
    sendResponse({ ok: true });

  } else if (msg.type === "MEET_LEFT") {
    if (state === State.CAPTURING || state === State.STARTING) {
      stopCapture();
    }
    // Tell the in-page orchestrator to stop any local capture and tear down its
    // observer + shadow hosts (Meet is an SPA; the content script is not
    // reloaded when the user leaves the call).
    const leftTabId = sender.tab?.id ?? meetTabId;
    if (leftTabId) {
      chrome.tabs.sendMessage(leftTabId, { type: "WD_MEET_LEFT" }).catch(() => {});
    }
    meetTabId = null;
    transitionTo(State.IDLE);
    sendResponse({ ok: true });

  } else if (msg.type === "CAPTURE_STARTED") {
    transitionTo(State.CAPTURING);
    relayToContentScript(msg);
    sendResponse({ ok: true });

  } else if (msg.type === "CAPTURE_RECONNECTING") {
    // Capture continues with buffered audio while the offscreen document
    // reconnects to the daemon — show a distinct badge so the user can
    // tell "recording but daemon unreachable" from plain recording.
    chrome.action.setBadgeText({ text: "···" });
    chrome.action.setBadgeBackgroundColor({ color: "#FF9800" });
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
    relayWD("WD_ERROR", { message: msg.message });
    transitionTo(State.IDLE);
    sendResponse({ ok: true });

  } else if (msg.type === "CHUNK_TRANSCRIBED") {
    relayToContentScript(msg);
    relayWD("WD_CHUNK", {
      text: msg.text,
      startTime: msg.startTime,
      chunkIndex: msg.chunkIndex,
    });
    sendResponse({ ok: true });

  } else if (msg.type === "STATUS") {
    // Daemon status push (from offscreen) drives the in-bar button/timer.
    relayWD("WD_STATUS", {
      state: msg.state,
      elapsed: msg.elapsed,
      chunks: msg.chunks,
    });
    sendResponse({ ok: true });

  } else if (msg.type === "SETTINGS") {
    relayWD("WD_SETTINGS", { settings: msg.settings, devices: msg.devices });
    sendResponse({ ok: true });

  } else if (msg.type === "WD_START") {
    // From content script (page gesture). Forward to offscreen.
    if (sender.tab?.id) {
      meetTabId = sender.tab.id;
      meetTitle = msg.title || "";
      meetUrl = msg.url || "";
      forwardToOffscreen({ type: "WD_START", title: meetTitle, url: meetUrl });
    }
    sendResponse({ ok: true });

  } else if (
    msg.type === "WD_STOP" ||
    msg.type === "WD_STOP_MEETING" ||
    msg.type === "WD_GET_SETTINGS" ||
    msg.type === "WD_SET_SETTINGS"
  ) {
    if (sender.tab?.id) {
      meetTabId = sender.tab.id;
      forwardToOffscreen(msg);
    }
    sendResponse({ ok: true });

  } else if (msg.type === "WD_PCM") {
    // High-frequency audio relay; offscreen already exists by now.
    if (sender.tab?.id) {
      chrome.runtime.sendMessage(msg).catch(() => {});
    }
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

// In-flight createDocument promise. Concurrent callers (e.g. two Meet tabs both
// booting WD_GET_SETTINGS, or a fast get-settings+start) would otherwise each
// see contexts.length===0 and both call createDocument — the second rejects
// with "Only a single offscreen document may be created". Memoize the creation
// so everyone awaits the same promise.
let offscreenCreating = null;

async function ensureOffscreenDocument() {
  const contexts = await chrome.runtime.getContexts({
    contextTypes: ["OFFSCREEN_DOCUMENT"],
  });
  if (contexts.length > 0) return;

  if (!offscreenCreating) {
    offscreenCreating = chrome.offscreen
      .createDocument({
        url: "offscreen.html",
        reasons: ["USER_MEDIA"],
        justification: "Capture tab audio for transcription",
      })
      .catch((err) => {
        // Belt-and-suspenders: if a racing caller already created it, swallow
        // the single-document error; re-throw anything else.
        if (!/single offscreen document/i.test(String(err && err.message))) {
          throw err;
        }
      })
      .finally(() => {
        offscreenCreating = null;
      });
  }

  await offscreenCreating;
}

// --- Relay to content script ---

function relayToContentScript(msg) {
  if (meetTabId) {
    chrome.tabs.sendMessage(meetTabId, msg).catch(() => {});
  }
}

// Relay a WD_* message to the Meet tab's content orchestrator.
function relayWD(type, payload) {
  if (!meetTabId) return;
  chrome.tabs.sendMessage(meetTabId, { type, ...payload }).catch(() => {});
}

// Ensure the offscreen document exists, then forward a message to it.
async function forwardToOffscreen(msg) {
  try {
    await ensureOffscreenDocument();
    chrome.runtime.sendMessage(msg).catch(() => {});
  } catch (err) {
    console.error("Failed to forward to offscreen:", err);
    relayWD("WD_ERROR", { message: "Nie udało się połączyć z offscreen." });
  }
}

// --- Keyboard command (optional): toggle capture on the Meet tab ---

if (chrome.commands && chrome.commands.onCommand) {
  chrome.commands.onCommand.addListener((command) => {
    if (command !== "toggle-capture") return;
    if (meetTabId) {
      chrome.tabs.sendMessage(meetTabId, { type: "WD_COMMAND_TOGGLE" }).catch(() => {});
    }
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
