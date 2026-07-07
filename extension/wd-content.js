/**
 * wd-content.js — orchestrator content script (loaded LAST). Wires the injected
 * control, settings panel, and caption band to the service worker, and owns the
 * page-gesture screen/tab audio capture via getDisplayMedia.
 *
 * Audio path: getDisplayMedia -> AudioWorklet PCM -> SW (WD_PCM) -> offscreen
 * -> WebSocket -> daemon. Button/timer state is driven ONLY by daemon status.
 */

"use strict";

(function () {
  const WD = window.WD || {};
  const injector = WD.injector;
  const panel = WD.panel;
  const captions = WD.captions;
  const logic = WD.logic;

  if (!injector || !panel || !captions || !logic) {
    console.error("wd: modules missing — content orchestrator aborting");
    return;
  }

  // --- Local state (UI mirrors daemon; capture ownership is local) ---
  let daemonState = "idle"; // idle | recording | transcribing
  let lastElapsed = 0; // last elapsed seconds from a status push (for reinject)
  let liveCaptionsOn = false;
  let captionLines = [];
  let currentSettings = null;
  let localCapture = null; // {audioContext, stream, worklet} when WE capture
  let starting = false; // re-entrancy guard for startCapture (F4)
  let sessionOwned = false; // true once WE issued WD_START (tab OR mic-only)

  const meetingMeta = () => ({
    title: (document.title || "Google Meet").replace(" - Google Meet", "").trim(),
    url: location.href,
  });

  // --- Toast (tiny transient message, shadow-free is fine, short-lived) ---
  function toast(text) {
    try {
      const el = document.createElement("div");
      el.textContent = text;
      Object.assign(el.style, {
        position: "fixed",
        bottom: "80px",
        left: "50%",
        transform: "translateX(-50%)",
        zIndex: "2147483647",
        background: "rgba(32,33,36,0.95)",
        color: "#fff",
        padding: "10px 16px",
        borderRadius: "8px",
        fontFamily: "Roboto, Arial, sans-serif",
        fontSize: "13px",
        boxShadow: "0 4px 16px rgba(0,0,0,0.4)",
      });
      document.body.appendChild(el);
      setTimeout(() => el.remove(), 3500);
    } catch (err) {
      console.warn("wd: toast failed", err);
    }
  }

  // --- Capture (page gesture required, runs in content script) ---
  async function startCapture() {
    // Re-entrancy guard (F4): a double click or a WD_COMMAND_TOGGLE while the
    // getDisplayMedia picker is open must not open a second picker. sessionOwned
    // extends the guard to the mic-only (capture_tab=false) path, where there is
    // no localCapture and `starting` clears synchronously — without it a rapid
    // double-activation before the daemon's 'recording' status lands would fire
    // two WD_START messages (findings 5 & 7). sessionOwned is cleared on the
    // daemon idle status, WD_ERROR, and meet-left, re-opening the guard.
    if (starting || localCapture || sessionOwned) return;
    // Do not act on a default while settings are unknown (finding 3): with
    // currentSettings still null the capture_tab default would be `true` and a
    // capture_tab=false user's early click would pop the screen picker. Defer:
    // (re)request settings and bail so the click can be retried once they land.
    if (currentSettings === null) {
      chrome.runtime.sendMessage({ type: "WD_GET_SETTINGS" });
      toast("Ładowanie ustawień — spróbuj ponownie za chwilę");
      return;
    }
    starting = true;
    try {
      // Honor capture_tab (F9): when participant capture is off, do not open the
      // screen picker at all — start a mic-only meeting and let the daemon record
      // the local mic natively.
      const captureTab = currentSettings.capture_tab !== false;
      if (!captureTab) {
        const { title, url } = meetingMeta();
        sessionOwned = true;
        chrome.runtime.sendMessage({ type: "WD_START", title, url });
        return;
      }

      let stream;
      try {
        stream = await navigator.mediaDevices.getDisplayMedia({
          video: { preferCurrentTab: true },
          audio: true,
        });
      } catch (err) {
        console.warn("wd: getDisplayMedia denied", err);
        toast("Nagrywanie anulowane");
        injector.setButtonState("idle");
        return;
      }

      try {
        // We only need audio — drop the video track immediately.
        stream.getVideoTracks().forEach((t) => t.stop());

        if (!stream.getAudioTracks().length) {
          stream.getTracks().forEach((t) => t.stop());
          toast("Brak dźwięku — zaznacz „Udostępnij dźwięk kartę”");
          injector.setButtonState("idle");
          return;
        }

        const audioContext = new AudioContext({ sampleRate: 16000 });
        const source = audioContext.createMediaStreamSource(stream);
        await audioContext.audioWorklet.addModule(
          chrome.runtime.getURL("pcm-processor.js")
        );
        const worklet = new AudioWorkletNode(audioContext, "pcm-processor");
        source.connect(worklet);
        // Do NOT connect to destination — avoid echoing meeting audio locally.

        worklet.port.onmessage = (event) => {
          const f32 = event.data;
          // runtime messaging serializes via JSON, so ship a plain number array.
          chrome.runtime.sendMessage({ type: "WD_PCM", samples: Array.from(f32) });
        };

        // The user (or another audio track) ending the share stops recording.
        stream.getAudioTracks().forEach((track) => {
          track.onended = () => stopCapture();
        });

        localCapture = { audioContext, stream, worklet };

        const { title, url } = meetingMeta();
        sessionOwned = true;
        chrome.runtime.sendMessage({ type: "WD_START", title, url });
      } catch (err) {
        console.error("wd: startCapture failed", err);
        try { stream.getTracks().forEach((t) => t.stop()); } catch {}
        toast("Nie udało się rozpocząć nagrywania");
        injector.setButtonState("idle");
      }
    } finally {
      starting = false;
    }
  }

  function teardownLocalCapture() {
    if (!localCapture) return;
    const { audioContext, stream, worklet } = localCapture;
    localCapture = null;
    try { if (worklet) worklet.port.onmessage = null; } catch {}
    try { if (audioContext) audioContext.close(); } catch {}
    try { if (stream) stream.getTracks().forEach((t) => t.stop()); } catch {}
  }

  function stopCapture() {
    const owned = sessionOwned;
    sessionOwned = false;
    teardownLocalCapture(); // no-op for a mic-only or menu-bar meeting
    if (owned) {
      // We started this meeting (tab OR mic-only) — end it via WD_STOP.
      chrome.runtime.sendMessage({ type: "WD_STOP" });
    } else {
      // Meeting was started elsewhere (menu bar) — stop it at the daemon.
      chrome.runtime.sendMessage({ type: "WD_STOP_MEETING" });
    }
    // Optimistic instant feedback: the socket stays open, so the daemon's real
    // transcribing→idle status will correct this shortly.
    injector.setButtonState("transcribing");
    panel.setRecording(true);
  }

  // --- Main toggle (button + panel action share this) ---
  function onToggle() {
    if (daemonState === "idle") startCapture();
    else stopCapture();
  }

  // --- Settings patch flow ---
  function sendPatch(patch) {
    if (!patch || !Object.keys(patch).length) return;
    chrome.runtime.sendMessage({ type: "WD_SET_SETTINGS", patch });
  }

  const DEFAULT_SETTINGS = {
    recording_device: "",
    capture_mic: true,
    capture_tab: true,
    capture_screenshots: false,
    screenshot_displays: "all",
    live_captions: false,
    diarize: false,
    diarize_mode: "hybrid",
    recording_dir: "~/Desktop",
    recording_formats: ["txt"],
  };

  function onReset() {
    const patch = currentSettings
      ? logic.diffSettings(currentSettings, DEFAULT_SETTINGS)
      : DEFAULT_SETTINGS;
    sendPatch(patch);
  }

  function applyCaptionsVisibility() {
    const on = liveCaptionsOn && daemonState === "recording";
    captions.setVisible(on);
    if (on) captions.render(captionLines);
  }

  // --- Incoming SW messages ---
  chrome.runtime.onMessage.addListener((msg) => {
    if (!msg || typeof msg !== "object") return;

    if (msg.type === "WD_STATUS") {
      daemonState = msg.state || "idle";
      lastElapsed = typeof msg.elapsed === "number" ? msg.elapsed : lastElapsed;
      injector.setButtonState(daemonState, msg.elapsed);
      panel.setRecording(daemonState !== "idle");
      if (daemonState !== "recording") {
        // Daemon says we are no longer recording (idle from a menu-bar/remote
        // stop, or transcribing after the audio stream closed). Release any
        // local getDisplayMedia capture so the tab-audio share indicator goes
        // out and the worklet stops posting PCM to a closed session. Guarded:
        // no-ops if the self-stop path already tore it down.
        teardownLocalCapture();
        if (daemonState === "idle") {
          captionLines = [];
          sessionOwned = false; // meeting fully finished
        }
      }
      applyCaptionsVisibility();
    } else if (msg.type === "WD_SETTINGS") {
      currentSettings = msg.settings || currentSettings;
      if (currentSettings && "live_captions" in currentSettings) {
        liveCaptionsOn = !!currentSettings.live_captions;
      }
      panel.render(currentSettings, msg.devices);
      applyCaptionsVisibility();
    } else if (msg.type === "WD_CHUNK") {
      captionLines = logic.captionsReducer(captionLines, msg.text || "");
      if (liveCaptionsOn && daemonState === "recording") {
        captions.render(captionLines);
      }
    } else if (msg.type === "WD_ERROR") {
      // A daemon/offscreen error (rejected start, lost WS after reconnect
      // window) means recording is over. Tear down local capture so the tab
      // stays shared no longer, and reset daemonState so the local model
      // matches the painted idle button — otherwise the next click would stop
      // instead of start (state is the single source of truth).
      teardownLocalCapture();
      daemonState = "idle";
      sessionOwned = false;
      captionLines = [];
      toast(msg.message || "Błąd nagrywania");
      injector.setButtonState("idle");
      panel.setRecording(false);
      applyCaptionsVisibility();
    } else if (msg.type === "WD_COMMAND_TOGGLE") {
      // Keyboard shortcut. Note: starting needs a page gesture, so a
      // command-triggered START may be blocked by the browser — stop always works.
      onToggle();
    } else if (msg.type === "WD_MEET_LEFT") {
      onMeetLeft();
    } else if (msg.type === "WD_MEET_JOINED") {
      onMeetJoined();
    }
  });

  // --- Boot / lifecycle ---
  function reinject() {
    injector.injectControl(onToggle, () => panel.toggle());
    // Meet re-rendered the toolbar slot — a fresh idle button was built. Repaint
    // it with the last known daemon state so an active recording keeps its red
    // pulse/timer instead of flashing idle until the next status push.
    injector.setButtonState(daemonState, lastElapsed);
  }

  // Meet is an SPA: leaving a call does not reload the content scripts. Stop any
  // local capture and tear down the observer + shadow hosts so nothing keeps
  // running (or scanning the DOM) after the meeting UI is gone.
  function onMeetLeft() {
    if (localCapture) stopCapture();
    // Background also fully tears down the offscreen socket on MEET_LEFT; clear
    // ownership so a later re-join starts clean.
    sessionOwned = false;
    injector.teardown();
    panel.teardown();
    captions.teardown();
  }

  // Build the control, observer, and panel wiring, then hydrate settings. Also
  // opens the control connection so the daemon pushes status — lighting the
  // in-bar button for a menu-bar-started meeting.
  function initUi() {
    panel.init({ onPatch: sendPatch, onAction: onToggle, onReset });
    reinject();
    injector.startObserver(reinject);
    chrome.runtime.sendMessage({ type: "WD_GET_SETTINGS" });
  }

  // Re-join in the same tab: rebuild everything.
  function onMeetJoined() {
    initUi();
  }

  function boot() {
    initUi();
  }

  if (document.body) boot();
  else document.addEventListener("DOMContentLoaded", boot, { once: true });
})();
