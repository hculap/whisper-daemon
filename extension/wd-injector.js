/**
 * wd-injector.js — injects the Whisper control into Google Meet's bottom
 * toolbar: one pill grouping the settings chevron (left, like Meet's mic and
 * camera splits) and the record button, which grows into a live timer while
 * recording.
 *
 * Placement: inline in the main control row, right before the "present / share
 * screen" button (between camera and share). Meet's official
 * #browser-extension-*-buttons slots sit OUTSIDE the main row (start/center are
 * left of the mic, end is right of "leave"), so they read as a detached blob —
 * we insert into the row itself and fall back to the end slot only if the row
 * can't be located.
 *
 * A single MutationObserver re-inserts the control across Meet's SPA re-renders.
 * The hover tooltip is fixed-positioned outside the toolbar so it cannot be
 * clipped. Classic content script sharing the isolated-world scope; exposes
 * window.WD.injector.
 */

"use strict";

(function () {
  const WD = (window.WD = window.WD || {});

  const CONTROL_ID = "wd-control";
  const STYLE_ID = "wd-control-style";
  const TOOLTIP_ID = "wd-tooltip";
  // Fallback official slot (rightmost, after "leave") if the row isn't found.
  const FALLBACK_SLOT_ID = "browser-extension-end-buttons";
  const CHEVRON_TIP = "Ustawienia nagrywania";

  // Buttons that bracket our insertion point. Meet localises aria-labels, so
  // match a few languages loosely.
  const PRESENT_SELECTOR =
    'button[aria-label*="Udostępnij ekran" i],button[aria-label*="Zaprezentuj" i],' +
    'button[aria-label*="Present" i],button[aria-label*="Share screen" i],' +
    'button[aria-label*="screen" i]';
  const CAMERA_SELECTOR =
    'button[aria-label*="kamer" i],button[aria-label*="camera" i]';

  const icon = (...args) => (WD.icons ? WD.icons.icon(...args) : "");
  const FONT = WD.icons ? WD.icons.FONT : "Roboto, Arial, sans-serif";

  // Sized to Meet's 48px toolbar buttons; colours sit on its ~#1f1f1f bar.
  const C = `#${CONTROL_ID}`;
  const STYLES = `
    ${C} { display: inline-flex; align-items: center; flex: none; height: 48px;
      margin: 0 4px; vertical-align: middle; border-radius: 24px; background: #2B2C2F;
      font-family: ${FONT}; }
    ${C} button { -webkit-appearance: none; appearance: none; border: none; margin: 0;
      font: inherit; cursor: pointer; display: inline-flex; align-items: center;
      justify-content: center; box-sizing: border-box; }
    ${C} button:focus-visible { outline: 2px solid #A8C7FA; outline-offset: 2px; }
    ${C} .wd-chev { width: 36px; height: 48px; padding: 0 0 0 4px; border-radius: 24px 0 0 24px;
      background: transparent; color: #C4C7C5; transition: background .15s, color .15s; }
    ${C} .wd-chev:hover { background: rgba(255,255,255,.08); }
    ${C} .wd-chev .wd-ic { transition: transform .2s ease; }
    ${C}[data-menu="open"] .wd-chev { background: #44464A; color: #A8C7FA; }
    ${C}[data-menu="open"] .wd-chev .wd-ic { transform: rotate(180deg); }

    ${C} .wd-main { position: relative; height: 48px; min-width: 48px; padding: 0;
      border-radius: 24px; gap: 10px; background: #3C3D40; color: #E3E3E3;
      font-size: 15px; font-weight: 500; font-variant-numeric: tabular-nums;
      white-space: nowrap; transition: background .15s, color .15s; }
    ${C} .wd-main:hover { background: #46484B; }
    ${C} .wd-label { font-size: 14px; }

    ${C}[data-mode="starting"] .wd-main, ${C}[data-mode="saving"] .wd-main {
      padding: 0 18px 0 14px; cursor: default; }
    ${C}[data-mode="starting"] .wd-spin { color: #A8C7FA; }
    ${C}[data-mode="saving"] .wd-spin { color: #FCC934; }
    ${C}[data-mode="recording"] .wd-main, ${C}[data-mode="warning"] .wd-main {
      padding: 0 18px; background: #CC3A2F; color: #FFFFFF; }
    ${C}[data-mode="recording"] .wd-main:hover, ${C}[data-mode="warning"] .wd-main:hover { background: #D7473C; }
    ${C}[data-mode="paused"] .wd-main { padding: 0 18px 0 14px; gap: 8px; color: #FCC934;
      box-shadow: inset 0 0 0 2px #FCC934; }
    ${C}[data-mode="offline"] .wd-main { background: #333537; color: #A8ACB0; }

    ${C} .wd-dot { width: 10px; height: 10px; border-radius: 5px; background: #FFFFFF;
      box-shadow: 0 0 0 4px rgba(255,255,255,.24); animation: wd-pulse 1.6s ease-in-out infinite; }
    ${C} .wd-bars { display: inline-flex; align-items: center; gap: 2px; height: 16px; }
    ${C} .wd-bars i { display: block; width: 3px; height: 4px; border-radius: 2px;
      background: rgba(255,255,255,.85); transition: height .25s ease; }
    ${C} .wd-badge { position: absolute; top: 0; right: 0; width: 18px; height: 18px;
      border-radius: 9px; background: #FCC934; color: #241A00; font-size: 12px; font-weight: 700;
      display: flex; align-items: center; justify-content: center; box-shadow: 0 0 0 2px #1F1F1F; }
    ${C} .wd-spin { animation: wd-spin .9s linear infinite; }

    #${TOOLTIP_ID} { position: fixed; z-index: 2147483646; max-width: 240px; padding: 6px 10px;
      border-radius: 6px; background: #E3E3E3; color: #1F1F1F; font: 500 12px/16px ${FONT};
      text-align: center; pointer-events: none; opacity: 0; transition: opacity .12s ease; }
    #${TOOLTIP_ID}.wd-in { opacity: 1; }

    @keyframes wd-pulse { 0%, 100% { opacity: 1; } 50% { opacity: .45; } }
    @keyframes wd-spin { to { transform: rotate(360deg); } }
    @media (prefers-reduced-motion: reduce) {
      ${C} .wd-dot, ${C} .wd-spin { animation: none; }
      ${C} .wd-chev .wd-ic, ${C} .wd-bars i { transition: none; }
    }
  `;

  // Main-button content per mode; time and level are filled in place.
  const CONTENT = {
    idle: () => icon("rec", 24),
    offline: () => icon("recOff", 24),
    starting: () => `${icon("spinner", 20, { stroke: 2.6, cls: "wd-spin" })}<span class="wd-label">Łączenie…</span>`,
    saving: () => `${icon("spinner", 20, { stroke: 2.6, cls: "wd-spin" })}<span class="wd-label">Zapisywanie…</span>`,
    recording: () => '<span class="wd-dot"></span><span class="wd-time"></span>' +
      '<span class="wd-bars" aria-hidden="true"><i></i><i></i><i></i></span>',
    warning: () => '<span class="wd-dot"></span><span class="wd-time"></span>' +
      '<span class="wd-bars" aria-hidden="true"><i></i><i></i><i></i></span>' +
      '<span class="wd-badge" aria-hidden="true">!</span>',
    paused: () => `${icon("pause", 20, { stroke: 2.6 })}<span class="wd-time"></span>`,
  };
  const BAR_PEAKS = [7, 15, 10];

  let view = { mode: "idle", time: "00:00", label: "Rozpocznij nagrywanie", tooltip: "" };
  let menuOpen = false;
  let level = 0;
  let tooltipEl = null;
  let tooltipFor = null;

  function ensureStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = STYLES;
    (document.head || document.documentElement).appendChild(style);
  }

  /**
   * Locate the insertion point in the main control row: the toolbar "cell"
   * wrapping the present/share button, so we can insert our control just before
   * it (i.e. after the camera). Walk up from the present button until its parent
   * also contains the camera button — that parent is the row, and the current
   * node is present's cell.
   * @returns {{parent: Element, before: Element} | null}
   */
  function insertionFor(present, camera) {
    let cell = present;
    while (cell.parentElement && !cell.parentElement.contains(camera)) {
      cell = cell.parentElement;
    }
    const parent = cell.parentElement;
    // Bail if camera and present share the same cell (can't split cleanly) or we
    // walked all the way to <body>.
    if (!parent || cell.contains(camera) || parent === document.body) return null;
    return { parent, before: cell };
  }

  function findRowInsertion() {
    const viewport = { width: window.innerWidth, height: window.innerHeight };
    const visible = (button) => WD.logic.isMeetToolbarRect(button.getBoundingClientRect(), viewport);
    const presents = [...document.querySelectorAll(PRESENT_SELECTOR)].filter(visible);
    const cameras = [...document.querySelectorAll(CAMERA_SELECTOR)].filter(visible);
    let best = null;
    for (const present of presents) {
      for (const camera of cameras) {
        const spot = insertionFor(present, camera);
        if (!spot) continue;
        const p = present.getBoundingClientRect();
        const c = camera.getBoundingClientRect();
        const distance = Math.abs((p.left + p.width / 2) - (c.left + c.width / 2));
        if (distance > Math.max(600, viewport.width * 0.4)) continue;
        if (!best || distance < best.distance) best = { ...spot, distance };
      }
    }
    return best && { parent: best.parent, before: best.before };
  }

  // --- Tooltip ---

  function tooltipText(button) {
    return button.classList.contains("wd-chev") ? CHEVRON_TIP : view.tooltip;
  }

  function showTooltip(button) {
    if (menuOpen) return;
    const text = tooltipText(button);
    if (!text) return;
    if (!tooltipEl) {
      tooltipEl = document.createElement("div");
      tooltipEl.id = TOOLTIP_ID;
      tooltipEl.setAttribute("aria-hidden", "true");
      document.body.appendChild(tooltipEl);
    }
    tooltipFor = button;
    tooltipEl.textContent = text;
    const rect = button.getBoundingClientRect();
    const tip = tooltipEl.getBoundingClientRect();
    const left = Math.max(8, Math.min(rect.left + rect.width / 2 - tip.width / 2, window.innerWidth - tip.width - 8));
    tooltipEl.style.left = `${left}px`;
    tooltipEl.style.top = `${Math.max(8, rect.top - tip.height - 10)}px`;
    tooltipEl.classList.add("wd-in");
  }

  function hideTooltip() {
    tooltipFor = null;
    if (tooltipEl) tooltipEl.classList.remove("wd-in");
  }

  function wireTooltip(button) {
    button.addEventListener("mouseenter", () => showTooltip(button));
    button.addEventListener("focus", () => showTooltip(button));
    button.addEventListener("mouseleave", hideTooltip);
    button.addEventListener("blur", hideTooltip);
    button.addEventListener("pointerdown", hideTooltip);
  }

  // --- Control ---

  function buildControl(onToggle, onChevron) {
    const control = document.createElement("div");
    control.id = CONTROL_ID;

    const chevron = document.createElement("button");
    chevron.type = "button";
    chevron.className = "wd-chev";
    chevron.setAttribute("aria-label", CHEVRON_TIP);
    chevron.setAttribute("aria-haspopup", "dialog");
    chevron.innerHTML = icon("chevUp", 20, { stroke: 2 });

    const main = document.createElement("button");
    main.type = "button";
    main.className = "wd-main";

    main.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      if (typeof onToggle === "function") onToggle();
    });
    chevron.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      if (typeof onChevron === "function") onChevron();
    });
    wireTooltip(chevron);
    wireTooltip(main);

    // Chevron LEFT, record button RIGHT — mirrors Meet's camera + settings split.
    control.appendChild(chevron);
    control.appendChild(main);
    return control;
  }

  function paint(control) {
    const main = control.querySelector(".wd-main");
    const chevron = control.querySelector(".wd-chev");
    if (!main || !chevron) return;
    if (control.dataset.mode !== view.mode) {
      control.dataset.mode = view.mode;
      main.innerHTML = (CONTENT[view.mode] || CONTENT.idle)();
    }
    const time = main.querySelector(".wd-time");
    // A text write replaces the text node: a childList mutation that wakes the
    // body observer (layout read, and a reinject in the fallback slot). The
    // timer view refreshes four times a second, so write only real changes.
    if (time && time.textContent !== view.time) time.textContent = view.time;
    main.setAttribute("aria-label", view.label);
    main.setAttribute("aria-busy", String(view.mode === "starting" || view.mode === "saving"));
    control.dataset.menu = menuOpen ? "open" : "closed";
    chevron.setAttribute("aria-expanded", String(menuOpen));
    paintLevel(main);
    if (tooltipFor === main) showTooltip(main);
  }

  function paintLevel(main) {
    main.querySelectorAll(".wd-bars i").forEach((bar, i) => {
      bar.style.height = `${Math.round(4 + (BAR_PEAKS[i] - 4) * level)}px`;
    });
  }

  /**
   * Inject the control. Idempotent: returns the existing node if present.
   * Returns the control element, or null if no anchor was found.
   */
  function injectControl(onToggle, onChevron) {
    try {
      ensureStyle();
      const spot = findRowInsertion();
      const existing = document.getElementById(CONTROL_ID);
      const viewport = { width: window.innerWidth, height: window.innerHeight };
      const control = existing || buildControl(onToggle, onChevron);
      if (spot) {
        if (control.parentElement !== spot.parent || control.nextElementSibling !== spot.before) {
          spot.parent.insertBefore(control, spot.before);
        }
      } else if (existing && WD.logic.isMeetToolbarRect(existing.getBoundingClientRect(), viewport)) {
        // A bottom fallback remains usable while Meet finishes rendering. The
        // observer keeps retrying so it can be promoted to the preferred row.
        return existing;
      } else {
        if (existing) existing.remove();
        // Meet has moved this nominal "end" slot into its page header in some
        // layouts. Append provisionally, then validate the rendered control —
        // an empty but valid flex slot may itself have zero width.
        const slot = document.getElementById(FALLBACK_SLOT_ID);
        if (!slot) return null;
        slot.appendChild(control);
        if (!WD.logic.isMeetToolbarRect(control.getBoundingClientRect(), viewport)) {
          control.remove();
          // Appending/removing the provisional control queues child-list
          // records synchronously.  Drop only those self-generated records so
          // the observer does not schedule the same rejected probe forever;
          // later Meet mutations remain observable.
          if (observer) observer.takeRecords();
          return null;
        }
      }
      paint(control);
      return control;
    } catch (err) {
      console.error("wd: injectControl failed", err);
      return null;
    }
  }

  let observer = null;

  function startObserver(reinject) {
    if (observer) return observer;
    let scheduled = false;
    const run = () => {
      scheduled = false;
      const control = document.getElementById(CONTROL_ID);
      const viewport = { width: window.innerWidth, height: window.innerHeight };
      // A surviving node can still be stranded in a recycled header container.
      // Re-resolve its anchor whenever it leaves the bottom toolbar band.
      const provisional = control && control.parentElement?.id === FALLBACK_SLOT_ID;
      if (!control || provisional || !WD.logic.isMeetToolbarRect(control.getBoundingClientRect(), viewport)) {
        try {
          reinject();
        } catch (err) {
          console.error("wd: reinject failed", err);
        }
      }
    };
    observer = new MutationObserver(() => {
      if (scheduled) return;
      scheduled = true;
      requestAnimationFrame(run);
    });
    observer.observe(document.body, { childList: true, subtree: true });
    return observer;
  }

  function repaint() {
    const control = document.getElementById(CONTROL_ID);
    if (control) paint(control);
  }

  /** @param {{mode: string, time: string, label: string, tooltip: string}} next from WD.logic.controlView */
  function setView(next) {
    view = { ...view, ...next };
    repaint();
  }

  function setMenuOpen(open) {
    menuOpen = !!open;
    if (menuOpen) hideTooltip();
    repaint();
  }

  /** @param {number} next 0..1 — the louder source's meter level. */
  function setLevel(next) {
    level = Math.max(0, Math.min(1, Number(next) || 0));
    const main = document.querySelector(`${C} .wd-main`);
    if (main) paintLevel(main);
  }

  function teardown() {
    try {
      if (observer) observer.disconnect();
    } catch (err) {
      console.warn("wd: observer disconnect failed", err);
    }
    observer = null;
    try {
      const control = document.getElementById(CONTROL_ID);
      if (control) control.remove();
      if (tooltipEl) tooltipEl.remove();
    } catch (err) {
      console.warn("wd: control remove failed", err);
    }
    tooltipEl = null;
    tooltipFor = null;
    menuOpen = false;
  }

  WD.injector = {
    CONTROL_ID,
    SLOT_ID: FALLBACK_SLOT_ID,
    injectControl,
    startObserver,
    setView,
    setMenuOpen,
    setLevel,
    teardown,
  };
})();
