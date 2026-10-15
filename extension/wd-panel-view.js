/**
 * wd-panel-view.js — markup and styles for the Whisper menu above the Meet
 * toolbar. Pure string builders over a model; wd-panel.js owns the host,
 * events and live updates.
 *
 * Model: {settings, devices, daemonState, starting, paused, elapsed, health,
 * warning, connection, title, diarization, subview, shortcut}.
 *
 * Classic content script sharing the isolated-world scope. Exposes WD.panelView.
 */

"use strict";

(function () {
  const WD = (window.WD = window.WD || {});
  const icon = (...args) => WD.icons.icon(...args);

  const FORMATS = ["txt", "srt", "vtt", "json"];
  const SOURCES = [
    { key: "mic", setting: "capture_mic", icon: "me", title: "Ty", sub: "Mikrofon" },
    { key: "browser", setting: "capture_tab", icon: "people", title: "Uczestnicy", sub: "Dźwięk z Meet" },
  ];
  const CONNECTION = {
    connected: { tone: "ok", text: "Połączono z daemonem" },
    unknown: { tone: "idle", text: "Łączenie z daemonem…" },
    offline: { tone: "bad", text: "Daemon offline", action: ["reconnect", "Połącz ponownie"] },
    unpaired: { tone: "warn", text: "Nie sparowano", action: ["pair", "Sparuj"] },
    rejected: { tone: "warn", text: "Klucz odrzucony", action: ["pair", "Sparuj ponownie"] },
  };

  const CSS = `
    :host { all: initial; }
    .wd-menu { position: fixed; z-index: 2147483646; width: 360px; max-height: 80vh;
      overflow-y: auto; overscroll-behavior: contain; box-sizing: border-box; padding: 8px 0;
      background: #282A2C; color: #E3E3E3; border-radius: 16px;
      box-shadow: 0 12px 32px rgba(0,0,0,.5), 0 2px 6px rgba(0,0,0,.35);
      font: 14px/20px var(--wd-font); display: none; user-select: none; }
    .wd-menu.wd-open { display: block; animation: wd-pop .14s ease-out; }
    .wd-menu::-webkit-scrollbar { width: 8px; }
    .wd-menu::-webkit-scrollbar-thumb { background: #5F6368; border-radius: 4px; }
    .wd-menu * { box-sizing: border-box; }
    button, select, input { font: inherit; color: inherit; }
    button { -webkit-appearance: none; appearance: none; border: none; margin: 0; cursor: pointer; }
    button:disabled { cursor: default; }
    button:focus-visible, select:focus-visible, input:focus-visible {
      outline: 2px solid #A8C7FA; outline-offset: 2px; }
    .wd-ic { flex: none; }

    .wd-head { padding: 12px 20px 16px; display: flex; flex-direction: column; gap: 12px; }
    .wd-head-row { display: flex; align-items: center; gap: 10px; min-height: 24px; }
    .wd-head-title { flex: 1; min-width: 0; font-size: 13px; color: #A8ACB0; text-align: right;
      white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .wd-chip { display: inline-flex; align-items: center; gap: 6px; height: 24px; padding: 0 10px 0 8px;
      border-radius: 12px; background: #333537; color: #C4C7C5; font-size: 12px; font-weight: 500;
      white-space: nowrap; }
    .wd-chip .wd-ic { margin-left: -2px; }
    .wd-chip-rec { background: #4A1C18; color: #FFB4AB; }
    .wd-chip-pause { background: #3A3220; color: #FFE08A; }
    .wd-timer { margin-top: -2px; font-size: 40px; line-height: 44px; letter-spacing: -.01em;
      font-variant-numeric: tabular-nums; }
    .wd-pair { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }

    .wd-btn { height: 40px; padding: 0 16px; border-radius: 20px; display: inline-flex;
      align-items: center; justify-content: center; gap: 8px; font-size: 14px; font-weight: 500;
      white-space: nowrap; transition: background .15s, filter .15s; }
    .wd-btn-tonal { background: #3C3D40; color: #E3E3E3; }
    .wd-btn-tonal:hover { background: #46484B; }
    .wd-btn-stop { background: #8C1D18; color: #FFDAD6; }
    .wd-btn-stop:hover { background: #9C2721; }
    .wd-btn-rec { height: 48px; border-radius: 24px; background: #CC3A2F; color: #FFFFFF; font-size: 15px; }
    .wd-btn-rec:hover:not(:disabled) { background: #D7473C; }
    .wd-btn-rec:disabled { background: #3C3D40; color: #C4C7C5; }
    .wd-btn-pair, .wd-btn-pair:hover:not(:disabled) { background: #A8C7FA; color: #062E6F; }
    .wd-kbd { margin-left: 4px; padding: 2px 7px; border-radius: 6px; background: rgba(255,255,255,.18);
      font-size: 12px; }
    .wd-btn-rec:disabled .wd-kbd { display: none; }

    .wd-div { height: 1px; margin: 4px 0; background: #3A3C3F; }
    .wd-label { padding: 12px 20px 6px; font-size: 12px; line-height: 16px; font-weight: 500;
      letter-spacing: .02em; color: #A8ACB0; }

    .wd-row { display: flex; align-items: center; gap: 16px; min-height: 56px; padding: 0 20px;
      width: 100%; background: none; text-align: left; }
    .wd-row > .wd-ic { color: #C4C7C5; }
    .wd-row.wd-tight { min-height: 44px; }
    .wd-click { cursor: pointer; }
    .wd-click:hover { background: rgba(255,255,255,.06); }
    .wd-text { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 2px; }
    .wd-grow { flex: 1; min-width: 0; }
    .wd-sub { font-size: 12px; line-height: 16px; color: #A8ACB0; }
    .wd-warn-text { color: #FCC934; }
    .wd-row.wd-silent > .wd-ic { color: #FCC934; }
    .wd-row.wd-silent [data-el="status"] { color: #FCC934; }

    .wd-meter { display: inline-flex; align-items: center; gap: 3px; height: 20px; flex: none; }
    .wd-meter i { display: block; width: 4px; height: 4px; border-radius: 2px; background: #5F6368;
      transition: height .25s ease; }
    .wd-row.wd-speaking .wd-meter i { background: #A8C7FA; }

    .wd-switch { position: relative; width: 52px; height: 32px; flex: none; padding: 0;
      border-radius: 16px; background: #36393B; box-shadow: inset 0 0 0 2px #8E918F; }
    .wd-switch .wd-thumb { position: absolute; top: 8px; left: 8px; width: 16px; height: 16px;
      border-radius: 12px; background: #8E918F; color: #A8C7FA; display: flex; align-items: center;
      justify-content: center; transition: all .15s ease; }
    .wd-switch[aria-checked="true"] { background: #A8C7FA; box-shadow: none; }
    .wd-switch[aria-checked="true"] .wd-thumb { top: 4px; left: 24px; width: 24px; height: 24px;
      background: #062E6F; }
    .wd-switch:disabled { opacity: .5; }

    .wd-tiles { padding: 0 16px 12px; display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }
    .wd-tile { height: 64px; padding: 0 14px; border-radius: 16px; background: #333537; color: #E3E3E3;
      display: flex; align-items: center; gap: 12px; text-align: left; transition: background .15s; }
    .wd-tile:hover { background: #3C3D40; }
    .wd-tile .wd-ic { color: #C4C7C5; }
    .wd-tile-title { display: block; font-weight: 500; white-space: nowrap; }
    .wd-tile-sub { display: block; font-size: 12px; line-height: 16px; color: #A8ACB0; }
    .wd-tile.wd-on { background: #004A77; color: #C2E7FF; }
    .wd-tile.wd-on:hover { background: #0B5586; }
    .wd-tile.wd-on .wd-ic, .wd-tile.wd-on .wd-tile-sub { color: #C2E7FF; }

    .wd-select { position: relative; display: inline-flex; align-items: center; flex: none; }
    .wd-select select { -webkit-appearance: none; appearance: none; height: 36px; max-width: 180px;
      padding: 0 32px 0 14px; border: none; border-radius: 18px; background: #333537; color: #E3E3E3;
      font-size: 13px; cursor: pointer; text-overflow: ellipsis; }
    .wd-select select:hover { background: #3C3D40; }
    .wd-select > .wd-ic { position: absolute; right: 10px; pointer-events: none; color: #C4C7C5; }
    .wd-select option { background: #282A2C; color: #E3E3E3; }
    .wd-field { padding: 0 20px 8px 58px; }
    .wd-field .wd-select { display: flex; }
    .wd-field select { width: 100%; max-width: none; height: 40px; border-radius: 10px; }

    .wd-seg { display: flex; flex: none; height: 32px; border-radius: 16px; overflow: hidden;
      box-shadow: inset 0 0 0 1px #5F6368; }
    .wd-seg button { display: flex; align-items: center; gap: 4px; padding: 0 12px; background: transparent;
      color: #C4C7C5; font-size: 12px; font-weight: 500; }
    .wd-seg button + button { box-shadow: inset 1px 0 0 #5F6368; }
    .wd-seg button[aria-pressed="true"] { padding-left: 10px; background: #004A77; color: #C2E7FF; }

    .wd-input { position: relative; display: inline-flex; align-items: center; flex: none; }
    .wd-input input { width: 170px; height: 36px; padding: 0 34px 0 14px; border: none; border-radius: 18px;
      background: #333537; color: #E3E3E3; font-size: 13px; text-overflow: ellipsis; }
    .wd-input input:hover { background: #3C3D40; }
    .wd-input .wd-ic { position: absolute; right: 12px; pointer-events: none; color: #C4C7C5; }

    .wd-chips { padding: 0 20px 10px 58px; display: flex; flex-wrap: wrap; gap: 6px; }
    .wd-chips button { height: 32px; padding: 0 12px; border-radius: 8px; background: transparent;
      box-shadow: inset 0 0 0 1px #5F6368; color: #C4C7C5; display: inline-flex; align-items: center;
      gap: 4px; font-size: 13px; font-weight: 500; }
    .wd-chips button[aria-pressed="true"] { padding-left: 8px; background: #004A77; box-shadow: none; color: #C2E7FF; }

    .wd-alert { margin: 4px 16px 8px; padding: 12px 14px; border-radius: 12px; background: #3A3220;
      display: flex; gap: 12px; user-select: text; }
    .wd-alert > .wd-ic { color: #FCC934; margin-top: 2px; }
    .wd-alert-body { display: flex; flex-direction: column; gap: 6px; min-width: 0; }
    .wd-alert-title { font-weight: 500; color: #FFE08A; }
    .wd-alert-text { font-size: 13px; line-height: 18px; color: #E6DCC3; }
    .wd-alert-actions { display: flex; gap: 8px; margin-top: 4px; }
    .wd-btn-sm { height: 32px; padding: 0 14px; border-radius: 16px; font-size: 13px; font-weight: 500; }
    .wd-btn-amber { background: #FCC934; color: #241A00; }
    .wd-btn-amber:hover { background: #FFD65C; }
    .wd-btn-ghost { background: transparent; color: #FFE08A; padding: 0 10px; }
    .wd-btn-ghost:hover { background: rgba(255,224,138,.1); }

    .wd-subhead { display: flex; align-items: center; gap: 8px; padding: 6px 16px 10px 8px; }
    .wd-icon-btn { width: 40px; height: 40px; flex: none; border-radius: 20px; background: transparent;
      color: #E3E3E3; display: flex; align-items: center; justify-content: center; }
    .wd-icon-btn:hover { background: rgba(255,255,255,.08); }

    .wd-empty { padding: 16px 20px; font-size: 13px; color: #A8ACB0; }

    .wd-foot { margin-top: 4px; min-height: 44px; padding: 0 12px 0 20px; border-top: 1px solid #3A3C3F;
      display: flex; align-items: center; gap: 8px; font-size: 12px; color: #A8ACB0; }
    .wd-foot-text { flex: 1; min-width: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .wd-foot-meta { padding-right: 8px; max-width: 150px; white-space: nowrap; overflow: hidden;
      text-overflow: ellipsis; }
    .wd-link { height: 32px; padding: 0 8px; border-radius: 16px; background: transparent; color: #A8C7FA;
      font-size: 12px; font-weight: 500; white-space: nowrap; }
    .wd-link:hover { background: rgba(168,199,250,.1); }
    .wd-led { width: 8px; height: 8px; border-radius: 4px; flex: none; background: #8E918F; }
    .wd-led-ok { background: #6DD58C; }
    .wd-led-warn { background: #FCC934; }
    .wd-led-bad { background: #FF8A80; }
    .wd-led-rec { background: #FF6B5E; }

    .wd-spin { animation: wd-spin .9s linear infinite; }
    @keyframes wd-spin { to { transform: rotate(360deg); } }
    @keyframes wd-pop { from { opacity: 0; transform: translateY(6px); } }
    @media (prefers-reduced-motion: reduce) {
      .wd-menu.wd-open, .wd-spin { animation: none; }
      .wd-switch .wd-thumb, .wd-meter i { transition: none; }
    }
  `;

  function esc(value) {
    return String(value ?? "").replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
  }

  const divider = () => '<div class="wd-div"></div>';
  const label = (text) => `<div class="wd-label">${esc(text)}</div>`;
  const meter = () => '<span class="wd-meter" aria-hidden="true"><i></i><i></i><i></i><i></i><i></i></span>';

  function switchButton(key, on, name) {
    return `<button type="button" class="wd-switch" role="switch" aria-checked="${!!on}" ` +
      `aria-label="${esc(name)}" data-switch="${key}"><span class="wd-thumb">` +
      `${on ? icon("check", 14, { stroke: 3 }) : ""}</span></button>`;
  }

  function selectControl(key, options, selected, name) {
    const opts = options.map(([value, text]) =>
      `<option value="${esc(value)}"${value === selected ? " selected" : ""}>${esc(text)}</option>`).join("");
    return `<span class="wd-select"><select data-key="${key}" aria-label="${esc(name)}">${opts}</select>` +
      `${icon("chevDown", 18, { stroke: 2 })}</span>`;
  }

  // --- Header ---

  function liveHeader(m) {
    const paused = !!m.paused;
    return `
      <div class="wd-head">
        <div class="wd-head-row">
          <span class="wd-chip ${paused ? "wd-chip-pause" : "wd-chip-rec"}">
            <span class="wd-led ${paused ? "wd-led-warn" : "wd-led-rec"}"></span>${paused ? "Pauza" : "Nagrywanie"}
          </span>
          <span class="wd-head-title">${esc(m.title)}</span>
        </div>
        <div class="wd-timer" data-el="timer" role="timer">${esc(WD.logic.formatElapsed(m.elapsed))}</div>
        <div class="wd-pair">
          <button type="button" class="wd-btn wd-btn-tonal" data-action="pause">
            ${paused ? icon("play", 18) : icon("pause", 18, { stroke: 2.6 })}${paused ? "Wznów" : "Pauza"}
          </button>
          <button type="button" class="wd-btn wd-btn-stop" data-action="toggle">
            ${icon("stop", 16)}Zakończ i zapisz
          </button>
        </div>
      </div>`;
  }

  function idleChip(m) {
    if (m.starting) return `<span class="wd-chip">${icon("spinner", 14, { stroke: 3, cls: "wd-spin" })}Łączenie…</span>`;
    if (m.daemonState === "transcribing") {
      return `<span class="wd-chip wd-chip-pause">${icon("spinner", 14, { stroke: 3, cls: "wd-spin" })}Zapisywanie…</span>`;
    }
    const link = CONNECTION[m.connection] || CONNECTION.unknown;
    const text = m.connection === "connected" ? "Gotowy" : link.text;
    return `<span class="wd-chip"><span class="wd-led wd-led-${link.tone}"></span>${esc(text)}</span>`;
  }

  function idleHeader(m) {
    if (m.connection === "unpaired" || m.connection === "rejected") {
      // No key, no recording: pairing is the only useful next step.
      return `
        <div class="wd-head">
          <div class="wd-head-row">${idleChip(m)}<span class="wd-head-title">${esc(m.title)}</span></div>
          <button type="button" class="wd-btn wd-btn-rec wd-btn-pair" data-action="pair">${icon("lock", 20)}Sparuj rozszerzenie</button>
        </div>`;
    }
    const busy = m.starting || m.daemonState === "transcribing";
    const text = m.starting ? "Łączenie…" : m.daemonState === "transcribing"
      ? "Zapisywanie poprzedniego…" : "Rozpocznij nagrywanie";
    const glyph = busy ? icon("spinner", 20, { stroke: 2.6, cls: "wd-spin" }) : icon("rec", 22);
    return `
      <div class="wd-head">
        <div class="wd-head-row">${idleChip(m)}<span class="wd-head-title">${esc(m.title)}</span></div>
        <button type="button" class="wd-btn wd-btn-rec" data-action="toggle"${busy ? " disabled" : ""}>
          ${glyph}${text}<span class="wd-kbd">${esc(m.shortcut)}</span>
        </button>
      </div>`;
  }

  // --- Sections ---

  function liveSources(m) {
    const s = m.settings || {};
    const rows = SOURCES.map((src) => {
      const off = s[src.setting] === false;
      return `
        <div class="wd-row" data-src="${src.key}">
          ${icon(src.icon)}
          <div class="wd-text">
            <span>${src.title}</span>
            <span class="wd-sub">${src.sub} · <span data-el="status">${off ? "wyłączone" : "…"}</span></span>
          </div>
          ${off ? "" : meter()}
        </div>`;
    }).join("");
    return label("Źródła") + rows + warningAlert(m.warning);
  }

  function warningAlert(warning) {
    if (!warning) return "";
    const action = warning.action
      ? `<button type="button" class="wd-btn-sm wd-btn-amber" data-action="warning-action">${esc(warning.action)}</button>`
      : "";
    return `
      <div class="wd-alert" role="alert">
        ${icon("warning", 20)}
        <div class="wd-alert-body">
          <span class="wd-alert-title">${esc(warning.title)}</span>
          <span class="wd-alert-text">${esc(warning.body)}</span>
          <div class="wd-alert-actions">${action}
            <button type="button" class="wd-btn-sm wd-btn-ghost" data-action="warning-dismiss">Ignoruj</button>
          </div>
        </div>
      </div>`;
  }

  function sourceSwitches(m) {
    const s = m.settings;
    const devices = [["", "Domyślny systemowy"], ...(m.devices || []).map((d) => [d, d])];
    return label("Źródła") + `
      <div class="wd-row">
        ${icon("me")}
        <div class="wd-text"><span>Ty</span><span class="wd-sub">Twój mikrofon</span></div>
        ${switchButton("capture_mic", s.capture_mic !== false, "Nagrywaj mnie")}
      </div>
      <div class="wd-field">${selectControl("recording_device", devices, s.recording_device || "", "Mikrofon")}</div>
      <div class="wd-row">
        ${icon("people")}
        <div class="wd-text"><span>Uczestnicy</span><span class="wd-sub">Dźwięk z Meet</span></div>
        ${switchButton("capture_tab", s.capture_tab !== false, "Nagrywaj uczestników")}
      </div>`;
  }

  function tile(key, glyph, title, on) {
    return `<button type="button" class="wd-tile${on ? " wd-on" : ""}" aria-pressed="${on}" data-tile="${key}">
      ${icon(glyph)}<span><span class="wd-tile-title">${title}</span>
      <span class="wd-tile-sub">${on ? "Włączone" : "Wyłączone"}</span></span></button>`;
  }

  function liveTiles(m) {
    const s = m.settings;
    return label("Na żywo") + `<div class="wd-tiles">
      ${tile("live_captions", "cc", "Napisy", !!s.live_captions)}
      ${tile("capture_screenshots", "shot", "Zrzuty ekranu", !!s.capture_screenshots)}
    </div>`;
  }

  function recordingSection(m) {
    const s = m.settings;
    const diar = m.diarization;
    const diarSub = diar && diar.available === false && diar.message
      ? `<span class="wd-sub wd-warn-text">${esc(diar.message)}</span>`
      : '<span class="wd-sub">Rozpoznawanie po zakończeniu</span>';
    const all = s.screenshot_displays !== "primary";
    return label("Nagranie") + `
      <div class="wd-row wd-tight">
        ${icon("lang")}<span class="wd-grow">Język</span>
        ${selectControl("recording_language", WD.logic.LANGUAGES, s.recording_language || "auto", "Język transkrypcji")}
      </div>
      <div class="wd-row">
        ${icon("diar")}<div class="wd-text"><span>Kto mówi</span>${diarSub}</div>
        ${switchButton("diarize", !!s.diarize, "Rozpoznawanie mówców")}
      </div>
      <div class="wd-row">
        ${icon("shot")}<div class="wd-text"><span>Ekrany</span><span class="wd-sub">Do zrzutów</span></div>
        <div class="wd-seg" role="group" aria-label="Ekrany do zrzutów">
          <button type="button" data-disp="all" aria-pressed="${all}">${all ? icon("check", 14, { stroke: 3 }) : ""}Wszystkie</button>
          <button type="button" data-disp="primary" aria-pressed="${!all}">${all ? "" : icon("check", 14, { stroke: 3 })}Główny</button>
        </div>
      </div>`;
  }

  function saveSection(m) {
    const s = m.settings;
    const formats = Array.isArray(s.recording_formats) ? s.recording_formats : ["txt"];
    const chips = FORMATS.map((f) => {
      const on = formats.includes(f);
      return `<button type="button" data-fmt="${f}" aria-pressed="${on}">${on ? icon("check", 14, { stroke: 3 }) : ""}${f}</button>`;
    }).join("");
    return label("Zapis") + `
      <div class="wd-row wd-tight">
        ${icon("folder")}<span class="wd-grow">Folder</span>
        <span class="wd-input"><input type="text" data-key="recording_dir" aria-label="Folder zapisu"
          spellcheck="false" value="${esc(s.recording_dir || "~/Desktop")}">${icon("pencil", 16)}</span>
      </div>
      <div class="wd-row wd-tight">${icon("doc")}<span class="wd-grow">Formaty</span></div>
      <div class="wd-chips" role="group" aria-label="Formaty zapisu">${chips}</div>`;
  }

  function settingsRow() {
    return `
      <button type="button" class="wd-row wd-click" data-action="settings">
        ${icon("sliders")}
        <span class="wd-text"><span>Ustawienia nagrania</span>
          <span class="wd-sub">Zmiany zadziałają od następnego</span></span>
        ${icon("chevRight", 18, { stroke: 2 })}
      </button>`;
  }

  function emptySettings(m) {
    const text = m.connection === "connected" || m.connection === "unknown"
      ? "Ładowanie ustawień…" : "Ustawienia pojawią się po połączeniu z daemonem.";
    return `<div class="wd-empty">${text}</div>`;
  }

  function footer(m, live) {
    const link = CONNECTION[m.connection] || CONNECTION.unknown;
    let right = "";
    if (link.action) {
      right = `<button type="button" class="wd-link" data-action="${link.action[0]}">${link.action[1]}</button>`;
    } else if (live && m.settings) {
      right = `<span class="wd-foot-meta">Zapis: ${esc(m.settings.recording_dir || "~/Desktop")}</span>`;
    } else if (m.settings) {
      right = '<button type="button" class="wd-link" data-action="reset">Przywróć domyślne</button>';
    }
    return `<div class="wd-foot"><span class="wd-led wd-led-${link.tone}"></span>
      <span class="wd-foot-text">${esc(link.text)}</span>${right}</div>`;
  }

  function settingsPage(m) {
    const body = m.settings
      ? [sourceSwitches(m), divider(), recordingSection(m), divider(), saveSection(m)].join("")
      : emptySettings(m);
    return `
      <div class="wd-subhead">
        <button type="button" class="wd-icon-btn" data-action="back" aria-label="Wróć">${icon("back", 20, { stroke: 2 })}</button>
        <div class="wd-text"><span>Ustawienia nagrania</span>
          <span class="wd-sub">Zmiany zadziałają od następnego nagrania</span></div>
      </div>
      ${divider()}${body}${footer(m, true)}`;
  }

  /** @returns {string} the whole menu for the model */
  function render(m) {
    const live = m.daemonState === "recording";
    if (live && m.subview === "settings") return settingsPage(m);
    if (live) {
      const tiles = m.settings ? divider() + liveTiles(m) : "";
      return [liveHeader(m), divider(), liveSources(m), tiles, divider(), settingsRow(), footer(m, true)].join("");
    }
    const body = m.settings
      ? [sourceSwitches(m), divider(), liveTiles(m), divider(), recordingSection(m), divider(), saveSection(m)].join("")
      : emptySettings(m);
    return [idleHeader(m), divider(), body, footer(m, false)].join("");
  }

  WD.panelView = { CSS, render, SOURCES };
})();
