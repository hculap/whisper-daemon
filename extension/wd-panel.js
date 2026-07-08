/**
 * wd-panel.js — settings menu rendered in a Shadow DOM, styled to match Google
 * Meet's native "more options" menu (dark surface, left-icon rows, dividers,
 * Material switches, Google Sans). Anchored above the Whisper control.
 *
 * Classic content script sharing the isolated-world scope. Exposes WD.panel.
 */

"use strict";

(function () {
  const WD = (window.WD = window.WD || {});

  const HOST_ID = "wd-panel-host";
  const FORMATS = ["txt", "srt", "vtt", "json"];

  // Outline icons (Google-Symbols-like) drawn with currentColor strokes; the
  // record glyph is a filled dot. Kept inline so nothing loads over the network.
  const ICONS = {
    rec: '<circle cx="12" cy="12" r="6" fill="currentColor" stroke="none"/>',
    mic: '<rect x="9" y="3" width="6" height="11" rx="3"/><path d="M6 11a6 6 0 0 0 12 0"/><path d="M12 17v4"/>',
    me: '<circle cx="12" cy="8" r="3.4"/><path d="M5 20a7 7 0 0 1 14 0"/>',
    people: '<circle cx="9" cy="8.5" r="3"/><path d="M3.2 19a5.8 5.8 0 0 1 11.6 0"/><path d="M15.5 6.2a3 3 0 0 1 0 5.6"/><path d="M17 13.4a5.8 5.8 0 0 1 3.8 5.6"/>',
    shot: '<rect x="3" y="6" width="18" height="13" rx="2.5"/><circle cx="12" cy="12.5" r="3.4"/>',
    cc: '<rect x="3" y="5" width="18" height="14" rx="2.5"/><path d="M7.5 13h3"/><path d="M13.5 13h3.5"/>',
    diar: '<circle cx="9" cy="9" r="3.2"/><path d="M3.5 20a5.5 5.5 0 0 1 11 0"/><path d="M16 8a3.6 3.6 0 0 1 0 8"/><path d="M18.4 6a6.8 6.8 0 0 1 0 12"/>',
    folder: '<path d="M3.5 7a2 2 0 0 1 2-2h3.2l2 2H18.5a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-13a2 2 0 0 1-2-2z"/>',
    doc: '<path d="M7 3.5h7l4 4V19a1.5 1.5 0 0 1-1.5 1.5h-9A1.5 1.5 0 0 1 6 19V5A1.5 1.5 0 0 1 7 3.5z"/><path d="M13.5 3.5V8H18"/><path d="M9 13h6"/><path d="M9 16h4"/>',
    reset: '<path d="M12 6V3L7.5 7.5 12 12V9a4.5 4.5 0 1 1-4.5 4.5"/>',
  };
  function ic(name) {
    return `<svg class="wd-ic" viewBox="0 0 24 24" aria-hidden="true">${ICONS[name] || ""}</svg>`;
  }

  const CSS = `
    :host { all: initial; }
    .wd-menu {
      position: fixed;
      z-index: 2147483646;
      width: 340px;
      max-height: 78vh;
      overflow-y: auto;
      background: #282a2c;
      color: #e3e3e3;
      border-radius: 12px;
      box-shadow: 0 8px 28px rgba(0,0,0,.6), 0 1px 3px rgba(0,0,0,.4);
      padding: 8px 0;
      font-family: "Google Sans", "Roboto", Arial, sans-serif;
      font-size: 14px;
      display: none;
      user-select: none;
    }
    .wd-menu.wd-open { display: block; }
    .wd-menu::-webkit-scrollbar { width: 8px; }
    .wd-menu::-webkit-scrollbar-thumb { background: #5f6368; border-radius: 4px; }

    .wd-row {
      display: flex; align-items: center; gap: 16px;
      min-height: 44px; padding: 4px 20px; box-sizing: border-box;
    }
    .wd-row.wd-click { cursor: pointer; }
    .wd-row.wd-click:hover { background: rgba(255,255,255,.08); }
    .wd-ic { width: 22px; height: 22px; flex: none; fill: none;
      stroke: #c4c7c5; stroke-width: 1.7; stroke-linecap: round; stroke-linejoin: round; }
    .wd-label { flex: 1; min-width: 0; }
    .wd-sub-label { flex: 1; padding-left: 38px; color: #c4c7c5; font-size: 13px; }

    /* Primary record action — echoes Meet's top menu item. */
    .wd-primary { width: 100%; text-align: left; background: none; border: none;
      color: inherit; font: inherit; }
    .wd-primary .wd-ic { fill: #e3e3e3; stroke: none; }
    .wd-primary .wd-txt { display: flex; flex-direction: column; line-height: 1.25; }
    .wd-primary .wd-title { font-size: 14px; }
    .wd-primary .wd-desc { font-size: 12px; color: #9aa0a6; }
    .wd-primary.wd-rec .wd-ic { fill: #f28b82; }
    .wd-primary.wd-rec .wd-title { color: #f28b82; }

    .wd-div { height: 1px; background: rgba(255,255,255,.12); margin: 8px 0; }

    /* Material switch */
    .wd-sw { appearance: none; -webkit-appearance: none; position: relative;
      width: 34px; height: 20px; border-radius: 10px; background: #5f6368;
      cursor: pointer; transition: background .15s; flex: none; margin: 0; }
    .wd-sw::before { content: ""; position: absolute; top: 3px; left: 3px;
      width: 14px; height: 14px; border-radius: 50%; background: #e3e3e3;
      transition: transform .15s, background .15s; }
    .wd-sw:checked { background: #8ab4f8; }
    .wd-sw:checked::before { transform: translateX(14px); background: #202124; }

    .wd-sel, .wd-txt-in {
      background: #303134; color: #e3e3e3; border: 1px solid #5f6368;
      border-radius: 8px; padding: 7px 10px; font: inherit; font-size: 13px;
      width: 100%; box-sizing: border-box;
    }
    .wd-field { padding: 2px 20px 8px 58px; }
    .wd-sel-inline { max-width: 130px; width: auto; }

    .wd-radio { display: flex; align-items: center; gap: 10px; padding: 6px 20px 6px 58px;
      cursor: pointer; }
    .wd-radio:hover { background: rgba(255,255,255,.06); }
    .wd-radio input { accent-color: #8ab4f8; width: 16px; height: 16px; }

    .wd-chips { display: flex; gap: 8px; flex-wrap: wrap; padding: 4px 20px 8px 58px; }
    .wd-chip { display: inline-flex; align-items: center; gap: 6px; padding: 5px 12px;
      border: 1px solid #5f6368; border-radius: 16px; cursor: pointer; font-size: 13px;
      color: #c4c7c5; }
    .wd-chip.wd-on { background: #004a77; border-color: #004a77; color: #d3e3fd; }
    .wd-chip input { display: none; }

    .wd-note { padding: 0 20px 6px 58px; font-size: 12px; color: #9aa0a6; display: none; }
    .wd-note.wd-show { display: block; }

    .wd-reset .wd-ic { stroke: #8ab4f8; }
    .wd-reset .wd-label { color: #8ab4f8; }
  `;

  let hostEl = null;
  let shadow = null;
  let popEl = null;
  let onPatch = null;
  let onAction = null;
  let onReset = null;
  let currentSettings = {};
  let recordingActive = false;
  let lastDevices = [];
  let outsideClick = null;
  let renderPending = false;

  function buildHost() {
    if (hostEl) return;
    hostEl = document.createElement("div");
    hostEl.id = HOST_ID;
    shadow = hostEl.attachShadow({ mode: "open" });

    const style = document.createElement("style");
    style.textContent = CSS;
    shadow.appendChild(style);

    popEl = document.createElement("div");
    popEl.className = "wd-menu";
    shadow.appendChild(popEl);

    document.body.appendChild(hostEl);

    shadow.addEventListener("focusout", () => {
      if (!renderPending) return;
      setTimeout(() => {
        if (renderPending && !isEditingInPanel()) render(currentSettings, lastDevices);
      }, 0);
    });

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
    renderPending = false;
  }

  function emitPatch(patch) {
    if (typeof onPatch === "function" && patch && Object.keys(patch).length) {
      onPatch(patch);
    }
  }

  function coldNote(key) {
    const logic = WD.logic;
    const isCold = logic && typeof logic.isColdSetting === "function"
      ? logic.isColdSetting(key)
      : false;
    return recordingActive && isCold
      ? '<div class="wd-note wd-show">od następnego nagrania</div>'
      : "";
  }

  function isEditingInPanel() {
    if (!popEl || !popEl.classList.contains("wd-open") || !shadow) return false;
    const el = shadow.activeElement;
    if (!el) return false;
    const tag = el.tagName;
    return tag === "SELECT" || (tag === "INPUT" && el.type === "text");
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

  function toggleRow(icon, label, key, checked, cold) {
    return `
      <label class="wd-row wd-click">
        ${ic(icon)}<span class="wd-label">${label}</span>
        <input type="checkbox" class="wd-sw" data-key="${key}" ${checked ? "checked" : ""}>
      </label>
      ${cold ? coldNote(key) : ""}`;
  }

  function render(settings, devices) {
    buildHost();
    if (settings && typeof settings === "object") currentSettings = settings;
    if (Array.isArray(devices)) lastDevices = devices;
    if (isEditingInPanel()) {
      renderPending = true;
      return;
    }
    renderPending = false;
    devices = lastDevices;
    const s = currentSettings || {};
    const formats = Array.isArray(s.recording_formats) ? s.recording_formats : ["txt"];

    const chips = FORMATS.map((f) => {
      const on = formats.includes(f) ? " wd-on" : "";
      return `<label class="wd-chip${on}"><input type="checkbox" data-fmt="${f}" ${formats.includes(f) ? "checked" : ""}>${f}</label>`;
    }).join("");

    const status = recordingActive ? "Nagrywanie w toku" : "Gotowy do nagrywania";

    popEl.innerHTML = `
      <button class="wd-row wd-click wd-primary ${recordingActive ? "wd-rec" : ""}" data-role="action">
        ${ic("rec")}
        <span class="wd-txt">
          <span class="wd-title">${recordingActive ? "Zatrzymaj nagrywanie" : "Rozpocznij nagrywanie"}</span>
          <span class="wd-desc">${status}</span>
        </span>
      </button>

      <div class="wd-div"></div>

      <div class="wd-row"><span class="wd-ic-slot">${ic("mic")}</span><span class="wd-label">Mikrofon</span></div>
      <div class="wd-field"><select class="wd-sel" data-key="recording_device">${devicesOptions(devices, s.recording_device || "")}</select></div>
      ${coldNote("recording_device")}
      ${toggleRow("me", "Nagrywaj mnie (Ja)", "capture_mic", s.capture_mic, true)}
      ${toggleRow("people", "Nagrywaj uczestników", "capture_tab", s.capture_tab, true)}

      <div class="wd-div"></div>

      ${toggleRow("shot", "Zrzuty ekranu", "capture_screenshots", s.capture_screenshots, false)}
      <label class="wd-radio"><input type="radio" name="wd-disp" data-disp="all" ${s.screenshot_displays !== "primary" ? "checked" : ""}>Wszystkie ekrany</label>
      <label class="wd-radio"><input type="radio" name="wd-disp" data-disp="primary" ${s.screenshot_displays === "primary" ? "checked" : ""}>Tylko główny</label>
      ${coldNote("screenshot_displays")}
      ${toggleRow("cc", "Napisy na żywo", "live_captions", s.live_captions, false)}
      ${toggleRow("diar", "Diaryzacja (kto mówi)", "diarize", s.diarize, true)}
      <div class="wd-row"><span class="wd-sub-label">Tryb</span>
        <select class="wd-sel wd-sel-inline" data-key="diarize_mode">
          <option value="hybrid" ${s.diarize_mode === "hybrid" ? "selected" : ""}>hybrid</option>
          <option value="batch" ${s.diarize_mode === "batch" ? "selected" : ""}>batch</option>
          <option value="realtime" ${s.diarize_mode === "realtime" ? "selected" : ""}>realtime</option>
        </select>
      </div>
      ${coldNote("diarize_mode")}

      <div class="wd-div"></div>

      <div class="wd-row"><span class="wd-ic-slot">${ic("folder")}</span><span class="wd-label">Katalog zapisu</span></div>
      <div class="wd-field"><input type="text" class="wd-txt-in" data-key="recording_dir" value="${escapeAttr(s.recording_dir || "~/Desktop")}"></div>
      ${coldNote("recording_dir")}
      <div class="wd-row"><span class="wd-ic-slot">${ic("doc")}</span><span class="wd-label">Formaty</span></div>
      <div class="wd-chips">${chips}</div>
      ${coldNote("recording_formats")}

      <div class="wd-div"></div>

      <div class="wd-row wd-click wd-reset" data-role="reset">${ic("reset")}<span class="wd-label">Przywróć domyślne</span></div>
    `;

    // The icon in label-only rows uses a slot wrapper; normalise its class.
    popEl.querySelectorAll(".wd-ic-slot > .wd-ic").forEach((el) => el.classList.add("wd-ic"));
    wireInputs();
  }

  function wireInputs() {
    popEl.querySelectorAll("[data-key]").forEach((el) => {
      el.addEventListener("change", () => {
        const key = el.getAttribute("data-key");
        if (el.type === "checkbox") emitPatch({ [key]: el.checked });
        else emitPatch({ [key]: el.value });
      });
    });
    popEl.querySelectorAll("[data-disp]").forEach((el) => {
      el.addEventListener("change", () => {
        if (el.checked) emitPatch({ screenshot_displays: el.getAttribute("data-disp") });
      });
    });
    popEl.querySelectorAll("[data-fmt]").forEach((el) => {
      el.addEventListener("change", () => {
        const chosen = Array.from(popEl.querySelectorAll("[data-fmt]"))
          .filter((b) => b.checked)
          .map((b) => b.getAttribute("data-fmt"));
        emitPatch({ recording_formats: chosen });
      });
    });
    const actionBtn = popEl.querySelector('[data-role="action"]');
    if (actionBtn) actionBtn.addEventListener("click", () => { if (typeof onAction === "function") onAction(); });
    const resetBtn = popEl.querySelector('[data-role="reset"]');
    if (resetBtn) resetBtn.addEventListener("click", () => { if (typeof onReset === "function") onReset(); });
  }

  function position() {
    const control = document.getElementById(
      WD.injector ? WD.injector.CONTROL_ID : "wd-control"
    );
    const width = 340;
    if (!control) {
      popEl.style.left = "50%";
      popEl.style.bottom = "90px";
      popEl.style.transform = "translateX(-50%)";
      return;
    }
    const rect = control.getBoundingClientRect();
    let left = rect.left + rect.width / 2 - width / 2;
    left = Math.max(8, Math.min(left, window.innerWidth - width - 8));
    popEl.style.left = `${left}px`;
    popEl.style.bottom = `${Math.max(8, window.innerHeight - rect.top + 10)}px`;
    popEl.style.transform = "none";
  }

  function show() {
    buildHost();
    if (renderPending) render(currentSettings, lastDevices);
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
