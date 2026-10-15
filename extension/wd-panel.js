/**
 * wd-panel.js — the Whisper menu above the Meet toolbar, rendered in a Shadow
 * DOM and anchored to the in-bar control. Markup and styles live in
 * wd-panel-view.js; this file owns the host, input handling and live updates.
 *
 * The daemon pushes status and health once a second. Those ticks only patch the
 * timer and level meters in place; the menu is rebuilt only when something
 * structural changes, and never while a field in it is being edited.
 *
 * Classic content script sharing the isolated-world scope. Exposes WD.panel.
 */

"use strict";

(function () {
  const WD = (window.WD = window.WD || {});

  const HOST_ID = "wd-panel-host";
  const MENU_WIDTH = 360;
  const ACTIONS = {
    toggle: "onAction",
    pause: "onPause",
    reset: "onReset",
    reconnect: "onReconnect",
    pair: "onPair",
    "warning-action": "onWarningAction",
    "warning-dismiss": "onWarningDismiss",
  };
  // Attributes that identify a control across re-renders (focus restore).
  const FOCUS_KEYS = ["action", "switch", "tile", "disp", "fmt", "key"];

  const INITIAL_MODEL = Object.freeze({
    settings: null,
    devices: [],
    daemonState: "idle",
    starting: false,
    paused: false,
    elapsed: 0,
    health: null,
    warning: null,
    connection: "unknown",
    title: "",
    diarization: null,
    subview: "main",
    shortcut: "⌘⇧M",
  });

  let hostEl = null;
  let shadow = null;
  let popEl = null;
  let callbacks = {};
  let model = INITIAL_MODEL;
  let renderedKey = null;
  let renderPending = false;
  let docListeners = null;

  // Everything except the 1 Hz values (elapsed, health), which tick() patches.
  function structuralKey(m) {
    return JSON.stringify([m.settings, m.devices, m.daemonState, m.starting, m.paused, m.warning,
      m.connection, m.title, m.diarization, m.subview, m.shortcut]);
  }

  function buildHost() {
    if (hostEl) return;
    hostEl = document.createElement("div");
    hostEl.id = HOST_ID;
    shadow = hostEl.attachShadow({ mode: "open" });

    const style = document.createElement("style");
    style.textContent = WD.panelView.CSS;
    shadow.appendChild(style);

    popEl = document.createElement("div");
    popEl.className = "wd-menu";
    popEl.setAttribute("role", "dialog");
    popEl.setAttribute("aria-label", "Whisper — nagrywanie");
    popEl.style.setProperty("--wd-font", WD.icons.FONT);
    popEl.addEventListener("click", onClick);
    popEl.addEventListener("change", onChange);
    shadow.appendChild(popEl);
    shadow.addEventListener("focusout", () => {
      setTimeout(() => { if (renderPending && !isEditing()) render(); }, 0);
    });

    document.body.appendChild(hostEl);
  }

  function isEditing() {
    const el = shadow && shadow.activeElement;
    return !!el && (el.tagName === "SELECT" || el.tagName === "INPUT");
  }

  function call(name, ...args) {
    const fn = callbacks[name];
    if (typeof fn === "function") fn(...args);
  }

  // Optimistic: the switch flips now; the daemon's settings echo confirms it.
  function patch(change) {
    if (!model.settings || !change || !Object.keys(change).length) return;
    model = { ...model, settings: { ...model.settings, ...change } };
    render();
    call("onPatch", change);
  }

  function onClick(event) {
    const button = event.target.closest("button");
    if (!button || button.disabled) return;
    const { action, switch: key, tile, disp, fmt } = button.dataset;
    if (action === "settings" || action === "back") {
      model = { ...model, subview: action === "settings" ? "settings" : "main" };
      render();
      const first = popEl.querySelector("button");
      if (first) first.focus({ preventScroll: true });
    } else if (action) {
      call(ACTIONS[action]);
    } else if (key) {
      patch({ [key]: button.getAttribute("aria-checked") !== "true" });
    } else if (tile) {
      patch({ [tile]: button.getAttribute("aria-pressed") !== "true" });
    } else if (disp) {
      patch({ screenshot_displays: disp });
    } else if (fmt && model.settings) {
      const current = Array.isArray(model.settings.recording_formats) ? model.settings.recording_formats : ["txt"];
      const next = current.includes(fmt) ? current.filter((f) => f !== fmt) : [...current, fmt];
      if (next.length) patch({ recording_formats: next });
    }
  }

  function onChange(event) {
    const el = event.target;
    const key = el.dataset && el.dataset.key;
    if (!key || !model.settings) return;
    const value = el.tagName === "INPUT" ? el.value.trim() : el.value;
    if (!value && key === "recording_dir") return;
    if (model.settings[key] !== value) {
      el.blur();
      patch({ [key]: value });
    }
  }

  function focusSelector() {
    const el = shadow && shadow.activeElement;
    if (!el || !el.dataset) return null;
    const attr = FOCUS_KEYS.find((name) => el.dataset[name] !== undefined);
    return attr ? `[data-${attr}="${CSS.escape(el.dataset[attr])}"]` : null;
  }

  function render() {
    if (!popEl) return;
    if (isEditing()) {
      renderPending = true;
      return;
    }
    renderPending = false;
    renderedKey = structuralKey(model);
    const scroll = popEl.scrollTop;
    const focus = focusSelector();
    popEl.innerHTML = WD.panelView.render(model);
    popEl.scrollTop = scroll;
    if (focus) {
      const target = popEl.querySelector(focus);
      if (target) target.focus({ preventScroll: true });
    }
    tick();
    if (isOpen()) position();
  }

  // 1 Hz values: timer and level meters, patched in place.
  function tick() {
    if (!popEl) return;
    const timer = popEl.querySelector('[data-el="timer"]');
    if (timer) timer.textContent = WD.logic.formatElapsed(model.elapsed);
    for (const src of WD.panelView.SOURCES) {
      const row = popEl.querySelector(`[data-src="${src.key}"]`);
      if (!row || !row.querySelector(".wd-meter")) continue;
      const known = !!(model.health && model.health[src.key]);
      const level = WD.logic.sourceView(known ? model.health[src.key] : null);
      const status = row.querySelector('[data-el="status"]');
      if (status) status.textContent = known ? level.label : "…";
      row.classList.toggle("wd-speaking", level.status === "speaking");
      row.classList.toggle("wd-silent", known && level.status === "silent");
      row.querySelectorAll(".wd-meter i").forEach((bar, i) => {
        bar.style.height = `${level.bars[i]}px`;
      });
    }
  }

  /** Merge model fields; rebuilds only on a structural change. */
  function update(partial) {
    const next = { ...model, ...partial };
    // Leaving a recording returns to the main menu.
    if (next.daemonState !== "recording") next.subview = "main";
    model = next;
    if (!popEl) return;
    if (structuralKey(model) !== renderedKey) render();
    else tick();
  }

  function renderSettings(settings, devices) {
    update({
      settings: settings && typeof settings === "object" ? settings : model.settings,
      devices: Array.isArray(devices) ? devices : model.devices,
    });
  }

  function controlEl() {
    return document.getElementById(WD.injector ? WD.injector.CONTROL_ID : "wd-control");
  }

  function position() {
    const control = controlEl();
    if (!control) {
      popEl.style.left = `${Math.max(8, window.innerWidth / 2 - MENU_WIDTH / 2)}px`;
      popEl.style.bottom = "96px";
      popEl.style.maxHeight = "70vh";
      return;
    }
    const rect = control.getBoundingClientRect();
    let left = rect.left + rect.width / 2 - MENU_WIDTH / 2;
    left = Math.max(8, Math.min(left, window.innerWidth - MENU_WIDTH - 8));
    popEl.style.left = `${left}px`;
    popEl.style.bottom = `${Math.max(8, window.innerHeight - rect.top + 12)}px`;
    popEl.style.maxHeight = `${Math.max(240, rect.top - 20)}px`;
  }

  function isOpen() {
    return !!popEl && popEl.classList.contains("wd-open");
  }

  function listen() {
    const outsideClick = (e) => {
      const path = e.composedPath();
      const control = controlEl();
      if (path.includes(hostEl) || (control && path.includes(control))) return;
      hide();
    };
    const onKey = (e) => {
      if (e.key !== "Escape") return;
      e.stopPropagation();
      hide();
      const chevron = controlEl() && controlEl().querySelector(".wd-chev");
      if (chevron) chevron.focus();
    };
    const onResize = () => position();
    document.addEventListener("click", outsideClick, true);
    document.addEventListener("keydown", onKey, true);
    window.addEventListener("resize", onResize);
    docListeners = { outsideClick, onKey, onResize };
  }

  function unlisten() {
    if (!docListeners) return;
    document.removeEventListener("click", docListeners.outsideClick, true);
    document.removeEventListener("keydown", docListeners.onKey, true);
    window.removeEventListener("resize", docListeners.onResize);
    docListeners = null;
  }

  function show() {
    buildHost();
    if (isOpen()) return;
    if (renderPending || renderedKey === null || structuralKey(model) !== renderedKey) render();
    position();
    popEl.classList.add("wd-open");
    listen();
    const first = popEl.querySelector("button:not(:disabled)");
    if (first) first.focus({ preventScroll: true });
    call("onOpenChange", true);
  }

  function hide() {
    if (!isOpen()) return;
    popEl.classList.remove("wd-open");
    unlisten();
    if (model.subview !== "main") update({ subview: "main" });
    call("onOpenChange", false);
  }

  function toggle() {
    if (isOpen()) hide();
    else show();
  }

  function teardown() {
    unlisten();
    try {
      if (hostEl) hostEl.remove();
    } catch (err) {
      console.warn("wd: panel host remove failed", err);
    }
    hostEl = null;
    shadow = null;
    popEl = null;
    renderedKey = null;
    renderPending = false;
    model = INITIAL_MODEL;
  }

  WD.panel = {
    init(next) {
      callbacks = next || {};
      buildHost();
    },
    render: renderSettings,
    update,
    show,
    hide,
    toggle,
    isOpen,
    teardown,
  };
})();
