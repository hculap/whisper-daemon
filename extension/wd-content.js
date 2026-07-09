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
  let localCapture = null; // {mode, audioContext, stream, worklet} when WE capture
  let starting = false; // re-entrancy guard for startCapture (F4)
  let sessionOwned = false; // true once WE issued WD_START (tab OR mic-only)

  // --- RTC tap (MAIN-world rtc-tap.js) state ---
  // The primary capture_tab path taps Meet's WebRTC remote audio directly (no
  // getDisplayMedia, no picker, no output-device dependency). The MAIN-world
  // hook reports how many remote audio tracks it sees; 0 for too long means Meet
  // changed and we should fall back to sharing the tab.
  let remoteTrackCount = null; // last WD_RTC_STATE count (null = unknown yet)
  let rtcFallbackTimer = null; // fires the "no participant tracks" banner
  const RTC_NO_TRACKS_MS = 6000;

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

  // --- "No participant audio" banner (persistent, unlike toast) ---
  // The shared-tab audio can be dead silence if the user shared the wrong
  // surface or didn't tick "Share tab audio" — otherwise participants are
  // dropped and the transcript fills with hallucinated gibberish. Watch the
  // captured PCM's RMS and warn, clearly and persistently, when it stays silent.
  // A quiet opening is normal — a meeting routinely joins with everyone muted
  // for far longer than a few seconds. Only a *long* dead window is evidence of
  // a wrong/silent surface, so the alarm window is generous. Dead captures with
  // zero audio tracks are already caught synchronously at start.
  const TAB_SILENCE_MS = 25000;
  // Match the daemon's keep criterion (RMS_SILENCE_FLOOR): energy below this is
  // dropped as silence anyway, so it must not count as "audio heard".
  const TAB_SILENCE_RMS = 0.005;
  // Require ~1s of contiguous energy (4 × 256ms frames) before trusting the tab
  // carries real audio — a lone join/notification chime or ambient blip must not
  // permanently disarm the watch while the daemon still drops every chunk.
  const TAB_AUDIO_MIN_FRAMES = 4;
  let tabSilenceSince = 0;
  let tabAudioFrames = 0; // consecutive frames at/above the floor
  let tabAudioEverHeard = false; // set only after sustained energy — then quiet is just quiet
  let bannerEl = null;

  function showBanner(text, onClick) {
    if (!bannerEl) {
      bannerEl = document.createElement("div");
      Object.assign(bannerEl.style, {
        position: "fixed", bottom: "96px", left: "50%",
        transform: "translateX(-50%)", zIndex: "2147483647",
        // Amber advisory, not an alarming red stop-now banner: capture may well
        // be healthy (pre-speech silence is indistinguishable from a dead tab by
        // RMS alone), so this nudges the user to verify rather than to react.
        background: "rgba(249,171,0,0.96)", color: "#202124",
        padding: "10px 16px", borderRadius: "8px", maxWidth: "90vw",
        textAlign: "center", fontFamily: "Roboto, Arial, sans-serif",
        fontSize: "13px", boxShadow: "0 4px 16px rgba(0,0,0,0.4)",
      });
      document.body.appendChild(bannerEl);
    }
    bannerEl.textContent = text;
    // A click handler makes the banner an actionable fallback trigger (the
    // getDisplayMedia fallback needs a user gesture — a banner click qualifies).
    bannerEl.style.cursor = onClick ? "pointer" : "default";
    bannerEl.onclick = onClick || null;
  }
  function clearBanner() {
    tabSilenceSince = 0;
    if (rtcFallbackTimer) { clearTimeout(rtcFallbackTimer); rtcFallbackTimer = null; }
    if (bannerEl) { bannerEl.onclick = null; bannerEl.remove(); bannerEl = null; }
  }

  function resetSilenceWatch() {
    tabAudioEverHeard = false;
    tabAudioFrames = 0;
    clearBanner();
  }

  function checkTabSilence(f32) {
    // Only diagnose a dead capture: once *sustained* real audio has been heard
    // the tab is proven to carry participant sound and later quiet spells are
    // just nobody talking — do not flip the warning on during natural pauses.
    if (tabAudioEverHeard) return;
    let sum = 0;
    for (let i = 0; i < f32.length; i++) sum += f32[i] * f32[i];
    const rms = Math.sqrt(sum / f32.length);
    if (rms >= TAB_SILENCE_RMS) {
      // Above the daemon's floor — but require a contiguous run so a single
      // chime/blip can't latch the watch off while chunks are still dropped.
      if (++tabAudioFrames >= TAB_AUDIO_MIN_FRAMES) {
        tabAudioEverHeard = true;
        clearBanner();
      }
      return;
    }
    // A sub-floor frame breaks the run — sustained energy must be contiguous.
    tabAudioFrames = 0;
    const now = Date.now();
    if (!tabSilenceSince) tabSilenceSince = now;
    else if (now - tabSilenceSince > TAB_SILENCE_MS) {
      showBanner(
        "🔇 Nie słychać uczestników. Najczęstsza przyczyna: w Meet głośnik jest " +
        "ustawiony na słuchawki/urządzenie inne niż „Domyślny” — Chrome nie nagrywa " +
        "wtedy dźwięku karty. Ustaw głośnik Meet na „Domyślny”. Sprawdź też " +
        "„Udostępnij dźwięk karty”. Jeśli dźwięk działa, zignoruj."
      );
    }
  }

  // --- Capture (page gesture required, runs in content script) ---
  async function startCapture() {
    // Re-entrancy guard (F4): a double click or a WD_COMMAND_TOGGLE while a
    // start is in flight must not fire twice. sessionOwned extends the guard to
    // the mic-only (capture_tab=false) and RTC paths, where there is no picker
    // and `starting` clears synchronously — without it a rapid double-activation
    // before the daemon's 'recording' status lands would fire two WD_START
    // messages (findings 5 & 7). sessionOwned is cleared on the daemon idle
    // status, WD_ERROR, and meet-left, re-opening the guard.
    if (starting || localCapture || sessionOwned) return;
    // Do not act on a default while settings are unknown (finding 3): with
    // currentSettings still null the capture_tab default would be `true` and a
    // capture_tab=false user's early click would start participant capture.
    // Defer: (re)request settings and bail so the click can be retried once they
    // land.
    if (currentSettings === null) {
      chrome.runtime.sendMessage({ type: "WD_GET_SETTINGS" });
      toast("Ładowanie ustawień — spróbuj ponownie za chwilę");
      return;
    }
    starting = true;
    try {
      // Honor capture_tab (F9): when participant capture is off, start a mic-only
      // meeting and let the daemon record the local mic natively.
      const captureTab = currentSettings.capture_tab !== false;
      if (!captureTab) {
        const { title, url } = meetingMeta();
        sessionOwned = true;
        chrome.runtime.sendMessage({ type: "WD_START", title, url });
        return;
      }

      // PRIMARY capture_tab path: tap Meet's WebRTC remote audio directly via
      // the MAIN-world rtc-tap.js. No getDisplayMedia, no picker, no "share tab
      // audio" checkbox, and no output-device dependency (Bluetooth / setSinkId
      // sinks make Chrome's tab-audio capture deliver silence). The local mic is
      // a sender track and is never delivered to 'track', so it is excluded.
      // getDisplayMedia is kept as a manual fallback (see startDisplayMedia
      // Fallback) for the rare case where 0 remote tracks are detected.
      const { title, url } = meetingMeta();
      sessionOwned = true;
      remoteTrackCount = null;
      resetSilenceWatch(); // fresh session — re-arm the dead-capture watch
      localCapture = { mode: "rtc" };
      // Button-gesture-initiated: the START reaches rtc-tap.js inside the click,
      // so AudioContext.resume() is allowed to un-suspend.
      window.postMessage({ source: "wd", type: "WD_RTC_START" }, location.origin);
      chrome.runtime.sendMessage({ type: "WD_START", title, url });
    } finally {
      starting = false;
    }
  }

  // getDisplayMedia FALLBACK — used only when the RTC tap reports 0 remote audio
  // tracks (Meet changed / no receivers), invoked from a banner click so the
  // required user gesture is present. When resumeSession is true the daemon
  // session is already started (RTC path) — do not re-send WD_START, only swap
  // the PCM source. Preserves the no-audio-track guard and the AudioContext /
  // worklet path.
  async function startDisplayMediaCapture(opts) {
    const resumeSession = !!(opts && opts.resumeSession);
    let stream;
    try {
      // preferCurrentTab / selfBrowserSurface / systemAudio are TOP-LEVEL
      // DisplayMediaStreamOptions. Scoping the prompt to the current tab removes
      // the whole class of wrong-surface picks. NOTE: this does NOT guarantee
      // non-silent audio — a live-but-silent tab-audio track is the usual
      // failure (Meet speaker routed to a non-default device).
      stream = await navigator.mediaDevices.getDisplayMedia({
        video: true,
        audio: true,
        preferCurrentTab: true,
        selfBrowserSurface: "include",
        systemAudio: "include",
      });
    } catch (err) {
      console.warn("wd: getDisplayMedia denied", err);
      toast("Nagrywanie anulowane");
      return;
    }

    try {
      // The picker blocks on the OS UI for seconds; during that await the
      // session can be torn down (user stop, menu-bar/remote idle, WD_ERROR,
      // WD_MEET_LEFT). Those teardowns run while localCapture is null, so they
      // only flip sessionOwned=false and never see this not-yet-wired capture.
      // If the session ended while the picker was open, drop the freshly
      // acquired stream instead of wiring WD_PCM (and possibly WD_START) into a
      // dead/next session — which would leave an orphaned tab share with no UI
      // to stop it and block the next start (localCapture stuck non-null).
      if (resumeSession && !sessionOwned) {
        try { stream.getTracks().forEach((t) => t.stop()); } catch {}
        return;
      }

      // We only need audio — drop the video track immediately.
      stream.getVideoTracks().forEach((t) => t.stop());

      if (!stream.getAudioTracks().length) {
        stream.getTracks().forEach((t) => t.stop());
        toast("Brak dźwięku — zaznacz „Udostępnij dźwięk kartę”");
        return;
      }

      // Instrument the captured track so a future silent-capture is diagnosable
      // from the console.
      const audioTrack = stream.getAudioTracks()[0];
      try {
        console.info("wd: tab audio track", {
          muted: audioTrack.muted,
          readyState: audioTrack.readyState,
          settings: audioTrack.getSettings ? audioTrack.getSettings() : null,
        });
      } catch (err) {
        console.warn("wd: getSettings failed", err);
      }
      audioTrack.onmute = () =>
        console.warn("wd: tab audio MUTED — Meet speaker likely on a non-default device");
      audioTrack.onunmute = () => console.info("wd: tab audio unmuted");

      const audioContext = new AudioContext({ sampleRate: 16000 });
      const source = audioContext.createMediaStreamSource(stream);
      await audioContext.audioWorklet.addModule(
        chrome.runtime.getURL("pcm-processor.js")
      );
      const worklet = new AudioWorkletNode(audioContext, "pcm-processor");
      source.connect(worklet);
      // Do NOT connect to destination — avoid echoing meeting audio locally.

      resetSilenceWatch(); // fresh source — re-arm the dead-capture watch
      worklet.port.onmessage = (event) => {
        const f32 = event.data;
        checkTabSilence(f32); // warn if the shared tab has no participant audio
        // runtime messaging serializes via JSON, so ship a plain number array.
        chrome.runtime.sendMessage({ type: "WD_PCM", samples: Array.from(f32) });
      };

      // The user (or another audio track) ending the share stops recording.
      stream.getAudioTracks().forEach((track) => {
        track.onended = () => stopCapture();
      });

      localCapture = { mode: "tab", audioContext, stream, worklet };

      if (!resumeSession) {
        const { title, url } = meetingMeta();
        sessionOwned = true;
        chrome.runtime.sendMessage({ type: "WD_START", title, url });
      }
    } catch (err) {
      console.error("wd: startDisplayMediaCapture failed", err);
      try { stream.getTracks().forEach((t) => t.stop()); } catch {}
      toast("Nie udało się rozpocząć nagrywania");
      if (!resumeSession) injector.setButtonState("idle");
    }
  }

  // Switch a live RTC session over to getDisplayMedia (banner-click gesture).
  // Stops the RTC tap but keeps the already-started daemon session, then swaps
  // the PCM source so we never double-feed WD_PCM.
  async function startDisplayMediaFallback() {
    if (!localCapture || localCapture.mode !== "rtc") return;
    try { window.postMessage({ source: "wd", type: "WD_RTC_STOP" }, location.origin); } catch {}
    localCapture = null; // release RTC ownership before opening the picker
    clearBanner();
    await startDisplayMediaCapture({ resumeSession: true });
    // If the picker was cancelled / had no audio, localCapture stays null while
    // the daemon session keeps running on nothing — restore the RTC tap so the
    // session is not left silent.
    if (!localCapture && sessionOwned) {
      remoteTrackCount = null;
      resetSilenceWatch();
      localCapture = { mode: "rtc" };
      window.postMessage({ source: "wd", type: "WD_RTC_START" }, location.origin);
    }
  }

  // Watch the RTC tap's remote-track count during an active RTC session. Zero
  // tracks for RTC_NO_TRACKS_MS means Meet exposes no receivers we can tap —
  // surface a clickable banner that hands off to the getDisplayMedia fallback.
  function handleRtcState() {
    if (!localCapture || localCapture.mode !== "rtc") return;
    if (remoteTrackCount === 0) {
      if (!rtcFallbackTimer) {
        rtcFallbackTimer = setTimeout(() => {
          rtcFallbackTimer = null;
          if (localCapture && localCapture.mode === "rtc" && remoteTrackCount === 0) {
            showBanner(
              "🔇 Nie wykryto ścieżek audio uczestników — Meet mógł się zmienić. " +
              "Kliknij tutaj, aby użyć „Udostępnij kartę”.",
              startDisplayMediaFallback
            );
          }
        }, RTC_NO_TRACKS_MS);
      }
    } else if (typeof remoteTrackCount === "number" && remoteTrackCount > 0) {
      if (rtcFallbackTimer) { clearTimeout(rtcFallbackTimer); rtcFallbackTimer = null; }
      // Tracks appeared — drop the no-tracks banner (the RMS silence watch keeps
      // its own separate diagnosis running).
      if (bannerEl && bannerEl.onclick === startDisplayMediaFallback) clearBanner();
    }
  }

  function teardownLocalCapture() {
    clearBanner();
    if (!localCapture) return;
    const cap = localCapture;
    localCapture = null;
    // RTC tap path: tell the MAIN-world hook to stop (it disconnects source
    // nodes and closes its own AudioContext). Idempotent — a duplicate STOP is
    // a no-op there.
    if (cap.mode === "rtc") {
      try { window.postMessage({ source: "wd", type: "WD_RTC_STOP" }, location.origin); } catch {}
      return;
    }
    // getDisplayMedia path: close our AudioContext and stop the shared tracks.
    const { audioContext, stream, worklet } = cap;
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
    recording_language: "auto",
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

  // --- RTC tap channel (MAIN-world rtc-tap.js -> this isolated world) ---
  // rtc-tap.js can't reach chrome.runtime, so it posts PCM/state on the window.
  // Validate strictly (same-window, source marker, shape) — a page could
  // otherwise inject fake PCM into the user's own transcript.
  window.addEventListener("message", (event) => {
    if (event.source !== window) return;
    const data = event.data;
    if (!data || data.source !== "wd-rtc") return;
    if (data.type === "WD_RTC_PCM") {
      // Only relay while WE own an RTC session — ignore stray frames.
      if (!localCapture || localCapture.mode !== "rtc") return;
      if (!Array.isArray(data.samples) || data.samples.length === 0) return;
      checkTabSilence(Float32Array.from(data.samples)); // dead-capture watch
      // Same relay the worklet path uses; the array is already JSON-safe.
      chrome.runtime.sendMessage({ type: "WD_PCM", samples: data.samples });
    } else if (data.type === "WD_RTC_STATE") {
      remoteTrackCount = typeof data.tracks === "number" ? data.tracks : remoteTrackCount;
      handleRtcState();
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
