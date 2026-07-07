/**
 * wd-panel.js — settings popover rendered inside a Shadow DOM so none of its
 * styles leak into (or inherit from) Google Meet. Anchored above #wd-control.
 *
 * Classic content script sharing the isolated-world scope. Exposes WD.panel.
 */

"use strict";

(function () {
  const WD = (window.WD = window.WD || {});

  const HOST_ID = "wd-panel-host";
  const FORMATS = ["txt", "srt", "vtt", "json"];

  const CSS = `
    :host { all: initial; }
    .wd-pop {
      position: fixed;
      z-index: 2147483646;
      width: 320px;
      max-height: 70vh;
      overflow-y: auto;
      background: #202124;
      color: #e8eaed;
      border: 1px solid #3c4043;
      border-radius: 12px;
      box-shadow: 0 8px 32px rgba(0,0,0,0.5);
      padding: 16px;
      font-family: "Google Sans", Roboto, Arial, sans-serif;
      font-size: 13px;
      display: none;
    }
    .wd-pop.wd-open { display: block; }
    h3 { margin: 0 0 12px; font-size: 15px; font-weight: 500; }
    .row { display: flex; align-items: center; justify-content: space-between; margin: 10px 0; gap: 10px; }
    .row label { flex: 1; }
    .note { font-size: 11px; color: #9aa0a6; margin: 2px 0 8px; display: none; }
    .note.wd-show { display: block; }
    select, input[type="text"] {
      background: #303134; color: #e8eaed; border: 1px solid #5f6368;
      border-radius: 6px; padding: 5px 8px; font-size: 12px; max-width: 170px;
    }
    input[type="text"] { width: 170px; }
    .formats { display: flex; gap: 10px; flex-wrap: wrap; }
    .formats label { display: flex; align-items: center; gap: 4px; flex: none; }
    .sub { padding-left: 14px; }
    hr { border: none; border-top: 1px solid #3c4043; margin: 12px 0; }
    .actions { display: flex; flex-direction: column; gap: 8px; margin-top: 14px; }
    button {
      cursor: pointer; border: none; border-radius: 8px; padding: 9px 12px;
      font-size: 13px; font-weight: 500; font-family: inherit;
    }
    .wd-action { background: #8ab4f8; color: #202124; }
    .wd-action.wd-recording { background: #ea4335; color: #fff; }
    .wd-reset { background: transparent; color: #8ab4f8; border: 1px solid #5f6368; }
  `;

  let hostEl = null;
  let shadow = null;
  let popEl = null;
  let onPatch = null; // (patch) => void
  let onAction = null; // () => void  (start/stop toggle)
  let onReset = null; // () => void
  let currentSettings = {};
  let recordingActive = false;
  let lastDevices = [];
  let outsideClick = null; // document click handler (for teardown removal)

  function buildHost() {
    if (hostEl) return;
    hostEl = document.createElement("div");
    hostEl.id = HOST_ID;
    shadow = hostEl.attachShadow({ mode: "open" });

    const style = document.createElement("style");
    style.textContent = CSS;
    shadow.appendChild(style);

    popEl = document.createElement("div");
    popEl.className = "wd-pop";
    shadow.appendChild(popEl);

    document.body.appendChild(hostEl);

    outsideClick = (e) => {
      if (!popEl.classList.contains("wd-open")) return;
      if (e.composedPath().includes(hostEl)) return;
      const control = document.getElementById(
        WD.injector ? WD.injector.CONTROL_ID : "wd-control"
      );
      if (control && e.composedPath().includes(control)) return;
      hide();
    };
    document.addEventListener("click", outsideClick);
  }

  /**
   * Remove the panel host and its document-level click listener. Safe to call
   * repeatedly; a later render()/show() rebuilds the host on demand.
   */
  function teardown() {
    try {
      if (outsideClick) document.removeEventListener("click", outsideClick);
    } catch (err) {
      console.warn("wd: panel listener remove failed", err);
    }
    outsideClick = null;
    try {
      if (hostEl) hostEl.remove();
    } catch (err) {
      console.warn("wd: panel host remove failed", err);
    }
    hostEl = null;
    shadow = null;
    popEl = null;
  }

  function emitPatch(patch) {
    if (typeof onPatch === "function" && patch && Object.keys(patch).length) {
      onPatch(patch);
    }
  }

  function coldNote() {
    return recordingActive
      ? '<div class="note wd-show">od następnego nagrania</div>'
      : "";
  }

  function devicesOptions(devices, selected) {
    const opts = ['<option value="">Domyślne systemowe</option>'];
    for (const d of devices || []) {
      const sel = d === selected ? " selected" : "";
      opts.push(`<option value="${escapeAttr(d)}"${sel}>${escapeHtml(d)}</option>`);
    }
    return opts.join("");
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  }
  function escapeAttr(s) {
    return escapeHtml(s).replace(/"/g, "&quot;");
  }

  /**
   * Render the popover from a settings snapshot.
   * @param {object} settings
   * @param {string[]} devices
   */
  function render(settings, devices) {
    buildHost();
    if (settings && typeof settings === "object") currentSettings = settings;
    if (Array.isArray(devices)) lastDevices = devices;
    devices = lastDevices;
    const s = currentSettings || {};
    const formats = Array.isArray(s.recording_formats) ? s.recording_formats : ["txt"];

    const formatBoxes = FORMATS.map((f) => {
      const checked = formats.includes(f) ? " checked" : "";
      return `<label><input type="checkbox" data-fmt="${f}"${checked}>${f}</label>`;
    }).join("");

    popEl.innerHTML = `
      <h3>Whisper — ustawienia</h3>

      <div class="row">
        <label>Mikrofon</label>
        <select data-key="recording_device">${devicesOptions(devices, s.recording_device || "")}</select>
      </div>
      ${coldNote()}

      <div class="row">
        <label>Nagrywaj mnie (Ja)</label>
        <input type="checkbox" data-key="capture_mic" ${s.capture_mic ? "checked" : ""}>
      </div>
      ${coldNote()}
      <div class="row">
        <label>Nagrywaj uczestników</label>
        <input type="checkbox" data-key="capture_tab" ${s.capture_tab ? "checked" : ""}>
      </div>
      ${coldNote()}

      <hr>

      <div class="row">
        <label>Zrzuty ekranu</label>
        <input type="checkbox" data-key="capture_screenshots" ${s.capture_screenshots ? "checked" : ""}>
      </div>
      <div class="row sub">
        <label><input type="radio" name="wd-disp" data-disp="all" ${s.screenshot_displays !== "primary" ? "checked" : ""}> Wszystkie ekrany</label>
      </div>
      <div class="row sub">
        <label><input type="radio" name="wd-disp" data-disp="primary" ${s.screenshot_displays === "primary" ? "checked" : ""}> Tylko główny</label>
      </div>
      ${coldNote()}

      <hr>

      <div class="row">
        <label>Napisy na żywo</label>
        <input type="checkbox" data-key="live_captions" ${s.live_captions ? "checked" : ""}>
      </div>

      <div class="row">
        <label>Diaryzacja (kto mówi)</label>
        <input type="checkbox" data-key="diarize" ${s.diarize ? "checked" : ""}>
      </div>
      <div class="row sub">
        <label>Tryb</label>
        <select data-key="diarize_mode">
          <option value="hybrid" ${s.diarize_mode === "hybrid" ? "selected" : ""}>hybrid</option>
          <option value="batch" ${s.diarize_mode === "batch" ? "selected" : ""}>batch</option>
          <option value="realtime" ${s.diarize_mode === "realtime" ? "selected" : ""}>realtime</option>
        </select>
      </div>
      ${coldNote()}

      <hr>

      <div class="row">
        <label>Katalog zapisu</label>
        <input type="text" data-key="recording_dir" value="${escapeAttr(s.recording_dir || "~/Desktop")}">
      </div>
      ${coldNote()}

      <div class="row"><label>Formaty</label></div>
      <div class="formats">${formatBoxes}</div>
      ${coldNote()}

      <div class="actions">
        <button class="wd-action ${recordingActive ? "wd-recording" : ""}" data-role="action">
          ${recordingActive ? "ZATRZYMAJ NAGRYWANIE" : "ROZPOCZNIJ NAGRYWANIE"}
        </button>
        <button class="wd-reset" data-role="reset">Przywróć domyślne</button>
      </div>
    `;

    wireInputs();
  }

  function wireInputs() {
    // Boolean + string keys
    popEl.querySelectorAll("[data-key]").forEach((el) => {
      el.addEventListener("change", () => {
        const key = el.getAttribute("data-key");
        if (el.type === "checkbox") emitPatch({ [key]: el.checked });
        else emitPatch({ [key]: el.value });
      });
    });

    // Screenshot displays radios
    popEl.querySelectorAll("[data-disp]").forEach((el) => {
      el.addEventListener("change", () => {
        if (el.checked) emitPatch({ screenshot_displays: el.getAttribute("data-disp") });
      });
    });

    // Recording formats — collect all checked boxes into an array
    popEl.querySelectorAll("[data-fmt]").forEach((el) => {
      el.addEventListener("change", () => {
        const chosen = Array.from(popEl.querySelectorAll("[data-fmt]"))
          .filter((b) => b.checked)
          .map((b) => b.getAttribute("data-fmt"));
        emitPatch({ recording_formats: chosen });
      });
    });

    const actionBtn = popEl.querySelector('[data-role="action"]');
    if (actionBtn) {
      actionBtn.addEventListener("click", () => {
        if (typeof onAction === "function") onAction();
      });
    }
    const resetBtn = popEl.querySelector('[data-role="reset"]');
    if (resetBtn) {
      resetBtn.addEventListener("click", () => {
        if (typeof onReset === "function") onReset();
      });
    }
  }

  function position() {
    const control = document.getElementById(
      WD.injector ? WD.injector.CONTROL_ID : "wd-control"
    );
    if (!control) {
      popEl.style.left = "50%";
      popEl.style.bottom = "90px";
      popEl.style.transform = "translateX(-50%)";
      return;
    }
    const rect = control.getBoundingClientRect();
    const width = 320;
    let left = rect.left + rect.width / 2 - width / 2;
    left = Math.max(8, Math.min(left, window.innerWidth - width - 8));
    popEl.style.left = `${left}px`;
    popEl.style.bottom = `${Math.max(8, window.innerHeight - rect.top + 10)}px`;
    popEl.style.transform = "none";
  }

  function show() {
    buildHost();
    position();
    popEl.classList.add("wd-open");
  }
  function hide() {
    if (popEl) popEl.classList.remove("wd-open");
  }
  function toggle() {
    buildHost();
    if (popEl.classList.contains("wd-open")) hide();
    else show();
  }

  function setRecording(active) {
    const next = !!active;
    if (next === recordingActive) return;
    recordingActive = next;
    // Reflect start/stop label + cold-field notes without losing settings.
    if (popEl && currentSettings) render(currentSettings);
  }

  WD.panel = {
    init(callbacks) {
      callbacks = callbacks || {};
      onPatch = callbacks.onPatch || null;
      onAction = callbacks.onAction || null;
      onReset = callbacks.onReset || null;
      buildHost();
    },
    render,
    show,
    hide,
    toggle,
    setRecording,
    teardown,
  };
})();
