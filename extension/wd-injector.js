/**
 * wd-injector.js — injects a native-styled Whisper control into Google Meet's
 * bottom toolbar, sized and coloured to match Meet's own round buttons.
 *
 * Placement: inline in the main control row, right before the "present / share
 * screen" button (between camera and share). Meet's official
 * #browser-extension-*-buttons slots sit OUTSIDE the main row (start/center are
 * left of the mic, end is right of "leave"), so they read as a detached blob —
 * we insert into the row itself and fall back to the end slot only if the row
 * can't be located.
 *
 * A single MutationObserver re-inserts the control across Meet's SPA re-renders.
 * Classic content script sharing the isolated-world scope; exposes window.WD.injector.
 */

"use strict";

(function () {
  const WD = (window.WD = window.WD || {});

  const CONTROL_ID = "wd-control";
  const STYLE_ID = "wd-control-style";
  // Fallback official slot (rightmost, after "leave") if the row isn't found.
  const FALLBACK_SLOT_ID = "browser-extension-end-buttons";

  // Buttons that bracket our insertion point. Meet localises aria-labels, so
  // match a few languages loosely.
  const PRESENT_SELECTOR =
    'button[aria-label*="Udostępnij ekran" i],button[aria-label*="Zaprezentuj" i],' +
    'button[aria-label*="Present" i],button[aria-label*="Share screen" i],' +
    'button[aria-label*="screen" i]';
  const CAMERA_SELECTOR =
    'button[aria-label*="kamer" i],button[aria-label*="camera" i]';

  // Meet dark toolbar tokens (from --gm3-icon-button-filled-*): 3rem square,
  // 1.5rem corner → 48px circle. Colours picked to sit on the ~#1f1f1f bar.
  const STYLES = `
    #${CONTROL_ID} {
      display: inline-flex;
      align-items: center;
      gap: 2px;
      margin: 0 4px;
      vertical-align: middle;
      font-family: "Google Sans", "Roboto", Arial, sans-serif;
    }
    #${CONTROL_ID} .wd-btn {
      position: relative;
      height: 48px;
      border: none;
      padding: 0;
      cursor: pointer;
      background: #333537;
      color: #e3e3e3;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      border-radius: 24px;
      transition: background .15s ease;
      -webkit-appearance: none;
      appearance: none;
    }
    #${CONTROL_ID} .wd-main { width: 48px; }
    /* Chevron on the LEFT, darker like Meet's camera "video settings" split. */
    #${CONTROL_ID} .wd-chevron { width: 30px; background: #282a2c; }
    /* Material state layer: subtle white overlay on hover, matching Meet. */
    #${CONTROL_ID} .wd-main:hover { background: #3c3e41; }
    #${CONTROL_ID} .wd-chevron:hover { background: #313336; }
    #${CONTROL_ID} .wd-btn:active { background: #43454a; }
    #${CONTROL_ID} .wd-dot {
      width: 14px; height: 14px; border-radius: 50%;
      background: currentColor; display: block;
    }
    #${CONTROL_ID} .wd-main.wd-recording {
      background: #d93025; color: #fff;
    }
    #${CONTROL_ID} .wd-main.wd-recording:hover { background: #e2483d; }
    #${CONTROL_ID} .wd-main.wd-recording .wd-dot { animation: wd-pulse 1.6s ease-in-out infinite; }
    #${CONTROL_ID} .wd-main.wd-transcribing { background: #f9ab00; color: #202124; }
    #${CONTROL_ID} .wd-chevron svg { width: 20px; height: 20px; fill: currentColor; }
    @keyframes wd-pulse { 0%,100% { opacity: 1; } 50% { opacity: .35; } }
  `;

  const CHEVRON_SVG =
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 8.29 6.29 14l1.42 1.41L12 11.12l4.29 4.29L17.71 14z"/></svg>';

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
  function findRowInsertion() {
    const present = document.querySelector(PRESENT_SELECTOR);
    const camera = document.querySelector(CAMERA_SELECTOR);
    if (!present || !camera) return null;

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

  function buildControl(onToggle, onChevron) {
    const control = document.createElement("div");
    control.id = CONTROL_ID;

    const main = document.createElement("button");
    main.type = "button";
    main.className = "wd-btn wd-main";
    main.title = "Whisper — nagrywaj / zatrzymaj";
    main.setAttribute("aria-label", "Whisper toggle recording");
    main.innerHTML = '<span class="wd-dot"></span>';

    const chevron = document.createElement("button");
    chevron.type = "button";
    chevron.className = "wd-btn wd-chevron";
    chevron.title = "Ustawienia Whisper";
    chevron.setAttribute("aria-label", "Whisper settings");
    chevron.innerHTML = CHEVRON_SVG;

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

    // Chevron LEFT, record button RIGHT — mirrors Meet's camera + settings split.
    control.appendChild(chevron);
    control.appendChild(main);
    return control;
  }

  /**
   * Inject the control. Idempotent: returns the existing node if present.
   * Returns the control element, or null if no anchor was found.
   */
  function injectControl(onToggle, onChevron) {
    try {
      const existing = document.getElementById(CONTROL_ID);
      if (existing) return existing;

      ensureStyle();
      const control = buildControl(onToggle, onChevron);

      const spot = findRowInsertion();
      if (spot) {
        spot.parent.insertBefore(control, spot.before);
        return control;
      }
      // Fallback: the official end slot (still inside the toolbar region).
      const slot = document.getElementById(FALLBACK_SLOT_ID);
      if (slot) {
        slot.appendChild(control);
        return control;
      }
      return null;
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
      if (!document.getElementById(CONTROL_ID)) {
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

  /**
   * Reflect daemon state on the main button.
   * @param {"idle"|"recording"|"transcribing"} state
   * @param {number} [elapsed] seconds — shown as the button tooltip.
   */
  function setButtonState(state, elapsed) {
    const control = document.getElementById(CONTROL_ID);
    if (!control) return;
    const main = control.querySelector(".wd-main");
    if (!main) return;
    main.classList.remove("wd-recording", "wd-transcribing");
    if (state === "recording") main.classList.add("wd-recording");
    else if (state === "transcribing") main.classList.add("wd-transcribing");

    let title = "Whisper — nagrywaj / zatrzymaj";
    if (state === "recording") {
      const t = typeof elapsed === "number" && WD.logic
        ? WD.logic.formatElapsed(elapsed) : "";
      title = t ? `Whisper — nagrywanie ${t} (kliknij aby zatrzymać)`
                : "Whisper — nagrywanie (kliknij aby zatrzymać)";
    } else if (state === "transcribing") {
      title = "Whisper — transkrypcja…";
    }
    main.title = title;
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
    } catch (err) {
      console.warn("wd: control remove failed", err);
    }
  }

  WD.injector = {
    CONTROL_ID,
    SLOT_ID: FALLBACK_SLOT_ID,
    injectControl,
    startObserver,
    setButtonState,
    teardown,
  };
})();
