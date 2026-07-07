/**
 * wd-injector.js — injects the native-styled split control into Meet's official
 * extension slot (#browser-extension-center-buttons) and keeps it alive across
 * Meet's SPA re-renders via a single MutationObserver.
 *
 * Classic content script: shares the isolated-world scope with the other
 * wd-*.js files. Exposes its API on the shared `window.WD` namespace.
 */

"use strict";

(function () {
  const WD = (window.WD = window.WD || {});

  const SLOT_ID = "browser-extension-center-buttons";
  const CONTROL_ID = "wd-control";
  const NATIVE_MIC_SELECTOR =
    'button[aria-label*="mikrofon" i], button[aria-label*="microphone" i]';

  const STYLE_ID = "wd-control-style";
  const STYLES = `
    #${CONTROL_ID}.wd-split {
      display: inline-flex;
      align-items: stretch;
      gap: 1px;
      vertical-align: middle;
      margin: 0 4px;
      font-family: "Google Sans", Roboto, Arial, sans-serif;
    }
    #${CONTROL_ID} .wd-main,
    #${CONTROL_ID} .wd-chevron {
      border: none;
      cursor: pointer;
      color: #fff;
      background: #3c4043;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      transition: background 0.15s ease;
    }
    #${CONTROL_ID} .wd-main.wd-fallback {
      width: 40px;
      height: 40px;
      border-radius: 20px 4px 4px 20px;
      font-size: 16px;
    }
    #${CONTROL_ID} .wd-chevron {
      width: 22px;
      height: 40px;
      border-radius: 4px 20px 20px 4px;
      font-size: 13px;
      line-height: 1;
    }
    #${CONTROL_ID} .wd-main:hover,
    #${CONTROL_ID} .wd-chevron:hover { background: #4d5156; }
    #${CONTROL_ID} .wd-main.wd-recording {
      background: #ea4335;
      animation: wd-pulse 1.6s ease-in-out infinite;
    }
    #${CONTROL_ID} .wd-main.wd-transcribing { background: #f9ab00; }
    #${CONTROL_ID} .wd-dot {
      width: 10px; height: 10px; border-radius: 50%;
      background: currentColor; display: inline-block;
    }
    #${CONTROL_ID} .wd-timer {
      font-variant-numeric: tabular-nums;
      font-size: 11px; font-weight: 600; margin-left: 4px;
    }
    @keyframes wd-pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.55; }
    }
  `;

  function ensureStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = STYLES;
    (document.head || document.documentElement).appendChild(style);
  }

  function nativeMicClass() {
    try {
      const mic = document.querySelector(NATIVE_MIC_SELECTOR);
      return mic ? mic.className : "";
    } catch (err) {
      console.warn("wd: failed reading native mic class", err);
      return "";
    }
  }

  /**
   * Inject the split control into the Meet slot. Idempotent: returns the
   * existing node if already present. Returns null if the slot is absent.
   */
  function injectControl(onToggle, onChevron) {
    try {
      const existing = document.getElementById(CONTROL_ID);
      if (existing) return existing;

      const slot = document.getElementById(SLOT_ID);
      if (!slot) return null;

      ensureStyle();

      const control = document.createElement("div");
      control.id = CONTROL_ID;
      control.className = "wd-split";

      const main = document.createElement("button");
      // Clone the native mic button's class for a pixel-perfect match, then add
      // our own fallback class so it still looks right if the clone is missing.
      const cloned = nativeMicClass();
      main.className = `wd-main${cloned ? " " + cloned : " wd-fallback"}`;
      main.type = "button";
      main.title = "Whisper — nagrywaj / zatrzymaj";
      main.setAttribute("aria-label", "Whisper toggle recording");
      main.innerHTML = '<span class="wd-dot"></span><span class="wd-timer"></span>';

      const chevron = document.createElement("button");
      chevron.className = "wd-chevron";
      chevron.type = "button";
      chevron.textContent = "⌃";
      chevron.title = "Ustawienia Whisper";
      chevron.setAttribute("aria-label", "Whisper settings");

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

      control.appendChild(main);
      control.appendChild(chevron);
      slot.appendChild(control);
      return control;
    } catch (err) {
      console.error("wd: injectControl failed", err);
      return null;
    }
  }

  let observer = null;

  /**
   * Watch the document for the slot repopulating (Meet re-renders its toolbar)
   * and re-run `reinject`. Debounced to a microtask-ish rAF to avoid churn.
   */
  function startObserver(reinject) {
    if (observer) return observer;
    let scheduled = false;

    const run = () => {
      scheduled = false;
      if (!document.getElementById(CONTROL_ID) && document.getElementById(SLOT_ID)) {
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
   * Reflect daemon state on the main button (idle dot / red pulse / timer).
   * @param {"idle"|"recording"|"transcribing"} state
   * @param {number} [elapsed] seconds
   */
  function setButtonState(state, elapsed) {
    const control = document.getElementById(CONTROL_ID);
    if (!control) return;
    const main = control.querySelector(".wd-main");
    const timer = control.querySelector(".wd-timer");
    if (!main) return;

    main.classList.remove("wd-recording", "wd-transcribing");
    if (state === "recording") main.classList.add("wd-recording");
    else if (state === "transcribing") main.classList.add("wd-transcribing");

    if (timer) {
      const showTimer = state === "recording" && typeof elapsed === "number";
      timer.textContent =
        showTimer && WD.logic ? WD.logic.formatElapsed(elapsed) : "";
    }
  }

  /**
   * Tear down the injector: disconnect the observer (so it stops scanning
   * Meet's DOM after the user leaves the call) and remove the injected control.
   * Safe to call repeatedly.
   */
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
    SLOT_ID,
    CONTROL_ID,
    injectControl,
    startObserver,
    setButtonState,
    teardown,
  };
})();
