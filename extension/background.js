chrome.storage.local.setAccessLevel({accessLevel: "TRUSTED_CONTEXTS"});
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

const State = { IDLE: "idle", DETECTED: "detected", STARTING: "starting", CAPTURING: "capturing", TRANSCRIBING: "transcribing" };

let state = State.IDLE;
// The tab in a Meet call. It outlives any single recording in that call, so
// only leaving the call or closing the tab releases it.
let meetTabId = null;
// A menu-bar (native) recording belongs to no tab; any Meet tab may control it.
let nativeRecording = false;
let meetTitle = "";
let meetUrl = "";
// The popup's own tab capture (START_WITH_STREAM). Every other recording is
// stopped through the tab or daemon that owns it.
let popupCapture = false;
// Daemon link as last reported: unknown | connected | offline | unpaired | rejected.
let connection = "unknown";

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
    [State.TRANSCRIBING]: { text: "…", color: "#FF9800" },
  };
  const badge = badges[state];
  chrome.action.setBadgeText({ text: badge.text });
  chrome.action.setBadgeBackgroundColor({ color: badge.color });
}

// No recording in progress: back to the joined call, if any.
function settleIdle() {
  nativeRecording = false;
  popupCapture = false;
  transitionTo(meetTabId ? State.DETECTED : State.IDLE);
}

// --- Messages ---

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.target === "offscreen") return false;
  if (msg.type === "GET_PAIRING_SECRET") {
    // chrome.runtime is the only API an offscreen document has, so it asks the
    // service worker for the key. Only that document may read it.
    if (sender.id !== chrome.runtime.id || sender.tab ||
        sender.url !== chrome.runtime.getURL("offscreen.html")) {
      sendResponse({ secret: null });
      return;
    }
    chrome.storage.local.get("pairingSecret")
      .then(({ pairingSecret }) => sendResponse({ secret: pairingSecret || null }))
      .catch(() => sendResponse({ secret: null }));
    return true;
  }
  if (msg.type === "MEET_JOINED") {
    if (state === State.IDLE) {
      meetTabId = sender.tab?.id ?? null;
      meetTitle = msg.title || "";
      meetUrl = msg.url || "";
      transitionTo(State.DETECTED);

      chrome.notifications.create("meet-detected", {
        type: "basic",
        iconUrl: "icons/icon128.png",
        title: "Wykryto spotkanie",
        message: `${meetTitle || "Google Meet"} — kliknij, aby nagrać`,
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
    if (sender.tab?.id !== meetTabId) { sendResponse({ok: true}); return; }
    // Leaving the call ends the session: fully tear down the offscreen socket
    // (persistent control channel) regardless of state. Safe if idle — the
    // offscreen just closes the socket.
    stopCapture();
    // Tell the in-page orchestrator to stop any local capture and tear down its
    // observer + shadow hosts (Meet is an SPA; the content script is not
    // reloaded when the user leaves the call).
    const leftTabId = sender.tab?.id ?? meetTabId;
    if (leftTabId) {
      chrome.tabs.sendMessage(leftTabId, { type: "WD_MEET_LEFT" }).catch(() => {});
    }
    meetTabId = null;
    popupCapture = false;
    transitionTo(State.IDLE);
    sendResponse({ ok: true });

  } else if (msg.type === "CAPTURE_STARTED") {
    transitionTo(State.CAPTURING);
    if (meetTabId !== null && !popupCapture) {
      chrome.tabs.sendMessage(meetTabId, {
        type: "WD_CAPTURE_STARTED",
        elapsed: msg.elapsed,
      }).catch(() => {});
    }
    sendResponse({ ok: true });

  } else if (msg.type === "CAPTURE_RECONNECTING") {
    // Capture continues with buffered audio while the offscreen document
    // reconnects to the daemon — show a distinct badge so the user can
    // tell "recording but daemon unreachable" from plain recording.
    chrome.action.setBadgeText({ text: "···" });
    chrome.action.setBadgeBackgroundColor({ color: "#FF9800" });
    sendResponse({ ok: true });

  } else if (msg.type === "CAPTURE_STOPPED") {
    settleIdle();
    sendResponse({ ok: true });

  } else if (msg.type === "CONTROL_CLOSED") {
    // The daemon went away: a "saving" or native-recording state can no longer
    // be confirmed or ended by a status. The daemon decides on the next start.
    if (state === State.TRANSCRIBING || nativeRecording) settleIdle();
    setConnection("unknown");
    sendResponse({ ok: true });

  } else if (msg.type === "CONNECTION") {
    setConnection(msg.state);
    sendResponse({ ok: true });

  } else if (msg.type === 'RECOVERY_AVAILABLE') {
    chrome.notifications.create('audio-recovery', {type:'basic',iconUrl:'icons/icon128.png',
      title:'Końcówka nagrania do pobrania',message:msg.message});
    sendResponse({ok:true});
  } else if (msg.type === "CAPTURE_ERROR") {
    chrome.notifications.create("capture-error", {
      type: "basic",
      iconUrl: "icons/icon128.png",
      title: "Błąd nagrywania",
      message: msg.message || "Nie udało się nagrać dźwięku",
    });
    relayWD("WD_ERROR", { message: msg.message });
    popupCapture = false;
    transitionTo(State.IDLE);
    sendResponse({ ok: true });

  } else if (msg.type === "CHUNK_TRANSCRIBED") {
    relayWD("WD_CHUNK", {
      text: msg.text,
      startTime: msg.startTime,
      chunkIndex: msg.chunkIndex,
    });
    sendResponse({ ok: true });

  } else if (msg.type === "STATUS") {
    setConnection("connected");
    // Daemon status push (from offscreen) drives the in-bar button/timer.
    relayWD("WD_STATUS", {
      state: msg.state,
      elapsed: msg.elapsed,
      chunks: msg.chunks,
    });
    // Track the daemon's real phase (badge included): WD_STOP keeps the socket
    // open and does not leave CAPTURING by itself. A settings snapshot reports
    // the phase too, but must not override a start that is being set up.
    if (!(msg.snapshot && state === State.STARTING)) {
      if (msg.state === "idle") settleIdle();
      else if (msg.state === "transcribing") transitionTo(State.TRANSCRIBING);
      else if (msg.state === "recording") {
        nativeRecording = !msg.sessionId;
        transitionTo(State.CAPTURING);
      }
    }
    sendResponse({ ok: true });

  } else if (msg.type === "HEALTH") {
    setConnection("connected");
    relayWD("WD_HEALTH", {sources: msg.sources, paused: msg.paused});
    sendResponse({ok: true});
  } else if (msg.type === "CAPABILITIES") {
    relayWD("WD_CAPABILITIES", {diarization: msg.diarization});
    sendResponse({ok: true});
  } else if (msg.type === "SETTINGS") {
    setConnection("connected");
    // Global read-only snapshot: initialize every content surface without
    // granting any of them ownership of a recording.
    broadcastMeet({ type: "WD_SETTINGS", settings: msg.settings, devices: msg.devices });
    sendResponse({ ok: true });

  } else if (msg.type === "WD_START") {
    // From content script (page gesture). Forward to offscreen.
    // A previous meeting still being saved is not refused here: the daemon is
    // the authority and makes the start wait until it is free.
    if (sender.tab?.id) {
      if (!nativeRecording && meetTabId !== null && meetTabId !== sender.tab.id &&
          (state === State.CAPTURING || state === State.STARTING)) {
        sendResponse({ok: false, error: "Another tab is recording"}); return;
      }
      transitionTo(State.STARTING);
      meetTabId = sender.tab.id;
      meetTitle = msg.title || "";
      meetUrl = msg.url || "";
      forwardToOffscreen({ type: "WD_START", title: meetTitle, url: meetUrl });
    }
    sendResponse({ ok: true });

  } else if (msg.type === "WD_STOP") {
    // A trailing stop must NOT resurrect meetTabId after MEET_LEFT cleared it,
    // or the status relay would re-point at a left tab (F15). Only refresh the
    // pointer if we still have a live meeting; always forward the stop so the
    // daemon finalizes.
    if (sender.tab?.id !== meetTabId) { sendResponse({ok: false}); return; }
    forwardToOffscreen(msg);
    sendResponse({ ok: true });

  } else if (msg.type === 'WD_GET_SETTINGS') {
    forwardToOffscreen(msg);
    sendResponse({ok:true});
  } else if (
    msg.type === "WD_PAUSE_MEETING" ||
    msg.type === "WD_STOP_MEETING" ||
    msg.type === "WD_SET_SETTINGS"
  ) {
    if (!sender.tab || meetTabId === null || nativeRecording || state === State.IDLE ||
        state === State.DETECTED || sender.tab.id === meetTabId) {
      forwardToOffscreen(msg);
    }
    sendResponse({ ok: true });

  } else if (msg.type === "WD_PCM") {
    // High-frequency audio relay. Once the offscreen document is known to exist
    // relay synchronously (preserves frame order); before that, await creation
    // so the first frames aren't dropped in the boot gap (F11/F23).
    if (sender.tab?.id === meetTabId) {
      if (offscreenReady) {
        chrome.runtime.sendMessage({ ...msg, target: "offscreen" }).catch(() => {});
      } else {
        ensureOffscreenDocument()
          .then(() => {
            offscreenReady = true;
            return chrome.runtime.sendMessage({ ...msg, target: "offscreen" }).catch(() => {});
          })
          .catch(() => {});
      }
    }
    sendResponse({ ok: true });

  } else if (msg.type === "GET_STATE") {
    sendResponse({ state, meetTabId, meetTitle, meetUrl, connection });

  } else if (msg.type === "OPEN_POPUP") {
    // The Meet menu's "pair" link. openPopup is unavailable on older Chrome.
    Promise.resolve().then(() => chrome.action.openPopup()).catch(() => {});
    sendResponse({ ok: true });

  } else if (msg.type === "FOCUS_MEET") {
    // Popup "change in Meet": bring the call forward with its menu open.
    resolveMeetTab().then((tabId) => {
      if (!tabId) return;
      chrome.tabs.update(tabId, { active: true })
        .then((tab) => tab && chrome.windows?.update(tab.windowId, { focused: true }))
        .catch(() => {});
      chrome.tabs.sendMessage(tabId, { type: "WD_OPEN_MENU" }).catch(() => {});
    });
    sendResponse({ ok: true });

  } else if (msg.type === "START_WITH_STREAM") {
    // Popup obtained streamId (has user gesture for tabCapture)
    if (state !== State.CAPTURING) {
      meetTabId = msg.tabId || meetTabId;
      popupCapture = true;
      transitionTo(State.STARTING);
      startCaptureWithStream(msg.streamId);
    }
    sendResponse({ ok: true });

  } else if (msg.type === "STOP_RECORDING") {
    // Popup Stop. A Meet-started session is stopped by its tab, which tears
    // down its own audio and keeps the socket for the final status; a full
    // teardown here would strand that tab on "recording".
    if (!popupCapture && meetTabId) {
      // The tab acknowledges; an unreachable tab falls back to a full teardown.
      chrome.tabs.sendMessage(meetTabId, { type: "WD_REQUEST_STOP" }).catch(() => stopCapture());
    } else if (!popupCapture && nativeRecording) {
      forwardToOffscreen({ type: "WD_STOP_MEETING" });
    } else if (state === State.CAPTURING || state === State.STARTING) {
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
      target: "offscreen",
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
  // Reaches the offscreen document if it exists; a missing receiver rejects,
  // which we ignore (nothing to tear down).
  chrome.runtime.sendMessage({ type: "STOP_CAPTURE", target: "offscreen" }).catch(() => {});
}

// In-flight createDocument promise. Concurrent callers (e.g. two Meet tabs both
// booting WD_GET_SETTINGS, or a fast get-settings+start) would otherwise each
// see contexts.length===0 and both call createDocument — the second rejects
// with "Only a single offscreen document may be created". Memoize the creation
// so everyone awaits the same promise.
let offscreenCreating = null;
// Fast-path flag so the hot WD_PCM relay doesn't await getContexts per frame
// once the offscreen document exists. The document is never closed here, so
// this stays true for the life of the service worker.
let offscreenReady = false;

async function ensureOffscreenDocument() {
  const contexts = await chrome.runtime.getContexts({
    contextTypes: ["OFFSCREEN_DOCUMENT"],
  });
  if (contexts.length > 0) {
    offscreenReady = true;
    return;
  }

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
  offscreenReady = true;
}

function setConnection(next) {
  if (!next || next === connection) return;
  connection = next;
  broadcastMeet({ type: "WD_CONNECTION", state: connection });
}

// Daemon-wide facts (settings, connection) reach every open Meet tab.
function broadcastMeet(message) {
  chrome.tabs.query({ url: "https://meet.google.com/*" }).then((tabs) => {
    for (const tab of tabs) chrome.tabs.sendMessage(tab.id, message).catch(() => {});
  }).catch(() => {});
}

// Relay a WD_* message to the Meet tab's content orchestrator.
//
// meetTabId lives in the service worker's memory, which MV3 wipes on eviction.
// A long transcription (diarization can take minutes) easily outlasts the SW's
// ~30s idle timeout once PCM stops flowing at 'stop'. The daemon's terminal
// 'idle' status then arrives AFTER eviction — it wakes the SW, but meetTabId is
// back to null, so the status is dropped and the in-bar button is stranded on
// the yellow 'transcribing' state. Recover the tab by querying open Meet tabs.
async function relayWD(type, payload) {
  const tabId = await resolveMeetTab();
  if (!tabId) return;
  chrome.tabs.sendMessage(tabId, { type, ...payload }).catch(() => {});
}

async function resolveMeetTab() {
  if (meetTabId) return meetTabId;
  try {
    const tabs = await chrome.tabs.query({ url: "https://meet.google.com/*" });
    if (tabs && tabs.length) {
      const t = tabs.find((x) => x.active) || tabs[0];
      return t.id ?? null;
    }
  } catch (err) {
    console.warn("wd: resolveMeetTab failed", err);
  }
  return null;
}

// Ensure the offscreen document exists, then forward a message to it.
async function forwardToOffscreen(msg) {
  try {
    await ensureOffscreenDocument();
    chrome.runtime.sendMessage({ ...msg, target: "offscreen" }).catch(() => {});
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
    // Tab gone — close the persistent offscreen socket unconditionally.
    stopCapture();
    meetTabId = null;
    popupCapture = false;
    transitionTo(State.IDLE);
  }
});
