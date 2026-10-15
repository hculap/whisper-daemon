"use strict";
/**
 * Popup: pairing, start/stop for the current tab, live levels and transcript,
 * and the export of audio the daemon never received. Everything it shows comes
 * from the service worker (GET_STATE) and the offscreen document's broadcasts.
 */

// Own scope: wd-logic.js declares its helpers as globals of this page.
(() => {
  const { formatElapsed, sourceView, LANGUAGES } = window.WD.logic;
  const { icon } = window.WD.icons;
  const $ = (id) => document.getElementById(id);

  const LINK = {
    connected: ["ok", "Połączono"],
    unknown: ["idle", "Sprawdzam…"],
    offline: ["bad", "Daemon offline"],
    unpaired: ["warn", "Nie sparowano"],
    rejected: ["warn", "Klucz odrzucony"],
  };
  const PAIR_MESSAGES = {
    connected: ["Połączono z daemonem.", false],
    offline: ["Klucz zapisany, ale daemon nie odpowiada. Uruchom: whisper-daemon restart", true],
    rejected: ["Daemon odrzucił klucz. Uruchom polecenie ponownie i wklej nowy klucz.", true],
  };
  const LANGUAGE_NAMES = Object.fromEntries(LANGUAGES);
  const MAX_TRANSCRIPT_LINES = 4;
  const MIN_KEY_LENGTH = 32;
  const COPIED_MS = 1500;

  // Background states: idle | detected | starting | capturing | transcribing.
  let ui = {
    state: "idle",
    meetTabId: null,
    meetTitle: "",
    connection: "unknown",
    paired: false,
    pairing: false, // pairing view opened from the status pill or awaiting a new key
    settings: null,
    diarization: null,
    paused: false,
    elapsed: 0,
    recovery: false,
    activeTab: null,
    shortcut: "",
  };

  function set(partial) {
    ui = { ...ui, ...partial };
    render();
  }

  // --- Rendering ---

  function paintIcon(el) {
    el.innerHTML = icon(el.dataset.icon, Number(el.dataset.size) || 20,
      { stroke: Number(el.dataset.stroke) || 1.8, cls: el.dataset.spin ? "spin" : "" });
  }

  function setIcon(el, name, spin = false) {
    if (el.dataset.icon === name && !!el.dataset.spin === spin) return;
    el.dataset.icon = name;
    if (spin) el.dataset.spin = "1";
    else delete el.dataset.spin;
    paintIcon(el);
  }

  function hostOf(url) {
    try {
      return new URL(url).hostname;
    } catch {
      return "";
    }
  }

  function render() {
    const forcedPairing = !ui.paired || ui.connection === "unpaired" || ui.connection === "rejected";
    const pairing = forcedPairing || ui.pairing;
    const live = !pairing && ui.state === "capturing";
    const [tone, text] = ui.paired ? (LINK[ui.connection] || LINK.unknown) : LINK.unpaired;

    $("linkLed").className = `led led-${tone}`;
    $("linkText").textContent = text;
    $("linkPill").setAttribute("aria-expanded", String(pairing));
    $("linkPill").setAttribute("aria-label", `Połączenie: ${text}. Parowanie`);
    $("linkPill").disabled = forcedPairing;

    $("recovery").hidden = !ui.recovery;
    $("pairView").hidden = !pairing;
    $("offline").hidden = pairing || ui.connection !== "offline";
    $("readyView").hidden = pairing || live;
    $("liveView").hidden = !live;
    if (!pairing && !live) renderReady();
    if (live) renderLive();
  }

  function renderReady() {
    const onMeet = !!ui.meetTabId;
    const starting = ui.state === "starting";
    const saving = ui.state === "transcribing";
    const tab = ui.activeTab;

    $("targetTitle").textContent = onMeet ? ui.meetTitle || "Google Meet" : tab?.title || "Ta karta";
    $("targetSub").textContent = onMeet ? "Google Meet" : hostOf(tab?.url) || "Ta karta";
    setIcon(document.querySelector(".meeting-icon"), onMeet ? "camera" : "tab");

    setIcon($("recordGlyph"), starting || saving ? "spinner" : "rec", starting || saving);
    $("recordLabel").textContent = starting ? "Łączenie…" : saving ? "Zapisywanie…"
      : onMeet ? "Nagraj spotkanie" : "Nagraj tę kartę";
    $("recordBtn").disabled = starting || saving || !(onMeet || tab?.id);
    $("recordKbd").hidden = !(onMeet && ui.shortcut);
    $("recordKbd").textContent = ui.shortcut;

    const s = ui.settings;
    $("summary").hidden = !s;
    $("openMeet").hidden = !onMeet;
    if (!s) return;
    const formats = Array.isArray(s.recording_formats) && s.recording_formats.length ? s.recording_formats : ["txt"];
    $("sumMic").textContent = s.capture_mic === false ? "Wyłączone" : s.recording_device || "Mikrofon systemowy";
    $("sumTab").textContent = s.capture_tab === false ? "Wyłączone" : "Dźwięk karty";
    $("sumLang").textContent = LANGUAGE_NAMES[s.recording_language || "auto"] || s.recording_language;
    $("sumSave").textContent = `${formats.join(", ")} · ${s.recording_dir || "~/Desktop"}`;
  }

  function renderLive() {
    $("liveChip").classList.toggle("paused", ui.paused);
    $("liveLed").className = `led ${ui.paused ? "led-warn" : "led-rec"}`;
    $("liveState").textContent = ui.paused ? "Pauza" : "Nagrywanie";
    $("liveTitle").textContent = ui.meetTitle;
    $("timer").textContent = formatElapsed(ui.elapsed);
    $("timer").setAttribute("aria-label", `Czas nagrania ${formatElapsed(ui.elapsed)}`);
    setIcon($("pauseGlyph"), ui.paused ? "play" : "pause");
    $("pauseLabel").textContent = ui.paused ? "Wznów" : "Pauza";

    const diarize = !!ui.settings?.diarize;
    const unavailable = ui.diarization && ui.diarization.available === false;
    $("diarNote").hidden = !diarize;
    $("diarText").textContent = unavailable && ui.diarization.message
      ? ui.diarization.message : "Mówców rozpoznam po zakończeniu nagrania.";
  }

  function renderSources(sources) {
    document.querySelectorAll("[data-src]").forEach((row) => {
      const source = sources && sources[row.dataset.src];
      const level = sourceView(source);
      row.querySelector(".source-status").textContent = source ? level.label : "…";
      row.classList.toggle("speaking", level.status === "speaking");
      row.classList.toggle("silent", !!source && level.status === "silent");
      row.querySelectorAll(".meter i").forEach((bar, i) => {
        bar.style.height = `${level.bars[i]}px`;
      });
    });
  }

  function addTranscript(text) {
    if (!text) return;
    const box = $("transcript");
    $("transcriptEmpty")?.remove();
    const line = document.createElement("p");
    line.textContent = text;
    box.appendChild(line);
    while (box.children.length > MAX_TRANSCRIPT_LINES) box.firstElementChild.remove();
  }

  function showError(message) {
    $("error").textContent = message || "";
    $("error").hidden = !message;
  }

  function pairStatus(message, bad = false) {
    $("pairStatus").textContent = message;
    $("pairStatus").classList.toggle("bad", bad);
  }

  // --- Daemon and service-worker events ---

  let awaitingPair = false;

  function onConnection(state) {
    if (awaitingPair && PAIR_MESSAGES[state]) {
      awaitingPair = false;
      pairStatus(...PAIR_MESSAGES[state]);
      if (state === "connected") {
        set({ connection: state, pairing: false });
        return;
      }
    }
    set({ connection: state });
  }

  chrome.runtime.onMessage.addListener((msg) => {
    if (!msg || typeof msg !== "object") return;
    switch (msg.type) {
      case "STATUS": {
        // A settings snapshot must not cancel a start that is being set up.
        if (msg.snapshot && ui.state === "starting") break;
        const idle = ui.meetTabId ? "detected" : "idle";
        const state = msg.state === "recording" ? "capturing" : msg.state === "transcribing" ? "transcribing" : idle;
        if (ui.connection !== "connected") onConnection("connected");
        set({ state, elapsed: typeof msg.elapsed === "number" ? msg.elapsed : ui.elapsed,
          paused: state === "capturing" && ui.paused });
        break;
      }
      case "HEALTH":
        if (ui.connection !== "connected") onConnection("connected");
        set({ paused: !!msg.paused });
        renderSources(msg.sources);
        break;
      case "SETTINGS":
        if (ui.connection !== "connected") onConnection("connected");
        set({ settings: msg.settings || ui.settings });
        break;
      case "CONNECTION":
        onConnection(msg.state);
        break;
      case "CONTROL_CLOSED":
        set({ connection: "unknown" });
        break;
      case "CAPABILITIES":
        set({ diarization: msg.diarization || null });
        break;
      case "CHUNK_TRANSCRIBED":
        addTranscript(msg.text);
        break;
      case "CAPTURE_STARTED":
        set({ state: "capturing" });
        break;
      case "CAPTURE_STOPPED":
        set({ state: ui.meetTabId ? "detected" : "idle" });
        break;
      case "CAPTURE_ERROR":
        showError(msg.message);
        set({ state: ui.meetTabId ? "detected" : "idle" });
        refreshRecovery();
        break;
      case "RECOVERY_AVAILABLE":
        refreshRecovery();
        break;
    }
  });

  // --- Actions ---

  $("linkPill").addEventListener("click", () => set({ pairing: !ui.pairing }));

  async function copyText(button, text) {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      return;
    }
    const label = button.getAttribute("aria-label");
    setIcon(button, "check");
    button.setAttribute("aria-label", "Skopiowano");
    setTimeout(() => {
      setIcon(button, "copy");
      button.setAttribute("aria-label", label);
    }, COPIED_MS);
  }

  $("copyPair").addEventListener("click", (e) => copyText(e.currentTarget, $("pairCommand").textContent));
  $("copyRestart").addEventListener("click", (e) => copyText(e.currentTarget, "whisper-daemon restart"));

  async function savePairing() {
    const secret = $("pairSecret").value.trim();
    if (secret.length < MIN_KEY_LENGTH) {
      pairStatus("Wklej cały klucz z Terminala.", true);
      return;
    }
    await chrome.storage.local.set({ pairingSecret: secret });
    $("pairSecret").value = "";
    pairStatus("Klucz zapisany. Łączę…");
    awaitingPair = true;
    set({ paired: true, pairing: true, connection: "unknown" });
    chrome.runtime.sendMessage({ type: "WD_GET_SETTINGS" }).catch(() => {});
  }

  $("savePair").addEventListener("click", savePairing);
  $("pairSecret").addEventListener("keydown", (e) => {
    if (e.key === "Enter") savePairing();
  });

  function tabStreamId(tabId) {
    return new Promise((resolve, reject) => {
      chrome.tabCapture.getMediaStreamId({ targetTabId: tabId }, (id) => {
        if (chrome.runtime.lastError) reject(new Error(chrome.runtime.lastError.message));
        else if (!id) reject(new Error("Pusty identyfikator strumienia"));
        else resolve(id);
      });
    });
  }

  // The popup click is the user gesture tabCapture needs.
  $("recordBtn").addEventListener("click", async () => {
    showError("");
    const tabId = ui.meetTabId || ui.activeTab?.id;
    if (!tabId) return;
    set({ state: "starting" });
    try {
      const streamId = await tabStreamId(tabId);
      chrome.runtime.sendMessage({ type: "START_WITH_STREAM", streamId, tabId });
    } catch (err) {
      showError(`Nie udało się przechwycić karty: ${err.message}`);
      set({ state: ui.meetTabId ? "detected" : "idle" });
    }
  });

  $("pauseBtn").addEventListener("click", () => {
    chrome.runtime.sendMessage({ type: "WD_PAUSE_MEETING" }).catch(() => {});
    set({ paused: !ui.paused });
  });

  $("stopBtn").addEventListener("click", () => {
    chrome.runtime.sendMessage({ type: "STOP_RECORDING" }).catch(() => {});
    set({ state: "transcribing" });
  });

  $("openMeet").addEventListener("click", () => {
    chrome.runtime.sendMessage({ type: "FOCUS_MEET" }).catch(() => {});
    window.close();
  });

  // --- Recovery: audio the daemon never received ---

  async function refreshRecovery() {
    try {
      const state = await chrome.runtime.sendMessage({ target: "offscreen", type: "GET_RECOVERY" });
      set({ recovery: !!(state?.count || state?.pending) });
    } catch {
      // No offscreen document yet: nothing is buffered.
    }
  }

  $("exportRecovery").addEventListener("click", async () => {
    const result = await chrome.runtime.sendMessage({ target: "offscreen", type: "EXPORT_RECOVERY" });
    if (!result?.url) {
      showError(result?.error || "Brak bufora audio.");
      return;
    }
    const link = document.createElement("a");
    link.href = result.url;
    link.download = result.filename;
    link.click();
    $("clearRecovery").disabled = false;
  });

  $("clearRecovery").addEventListener("click", async () => {
    await chrome.runtime.sendMessage({ target: "offscreen", type: "CLEAR_EXPORTED_RECOVERY" });
    $("clearRecovery").disabled = true;
    refreshRecovery();
  });

  // --- Boot ---

  async function init() {
    document.querySelectorAll("[data-icon]").forEach(paintIcon);
    $("pairCommand").textContent = `whisper-daemon pair ${chrome.runtime.id}`;
    const [stored, state, tabs, commands] = await Promise.all([
      chrome.storage.local.get("pairingSecret"),
      chrome.runtime.sendMessage({ type: "GET_STATE" }).catch(() => null),
      chrome.tabs.query({ active: true, currentWindow: true }),
      chrome.commands.getAll().catch(() => []),
    ]);
    set({
      paired: !!stored.pairingSecret,
      state: state?.state || "idle",
      meetTabId: state?.meetTabId ?? null,
      meetTitle: state?.meetTitle || "",
      connection: state?.connection || "unknown",
      activeTab: tabs[0] || null,
      shortcut: commands.find((c) => c.name === "toggle-capture")?.shortcut || "",
    });
    // Also probes the daemon link: the answer is SETTINGS or a CONNECTION state.
    if (stored.pairingSecret) chrome.runtime.sendMessage({ type: "WD_GET_SETTINGS" }).catch(() => {});
    refreshRecovery();
  }

  init();
})();
