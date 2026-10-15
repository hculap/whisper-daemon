/**
 * wd-toast.js — short-lived snackbar above the Meet toolbar, in its own Shadow
 * DOM so Meet's styles cannot reach it. One toast at a time; a new one replaces
 * the current one.
 *
 * Classic content script sharing the isolated-world scope. Exposes WD.toast.
 */

"use strict";

(function () {
  const WD = (window.WD = window.WD || {});

  const HOST_ID = "wd-toast-host";
  const VISIBLE_MS = 4000;

  const CSS = `
    :host { all: initial; }
    .wd-toast {
      position: fixed; left: 50%; bottom: 96px; z-index: 2147483647;
      display: flex; align-items: center; gap: 10px;
      max-width: min(520px, calc(100vw - 32px)); box-sizing: border-box;
      padding: 12px 16px; border-radius: 8px;
      background: #E3E3E3; color: #1F1F1F;
      font: 14px/20px var(--wd-font); letter-spacing: .01em;
      box-shadow: 0 6px 20px rgba(0,0,0,.45);
      opacity: 0; transform: translate(-50%, 8px);
      transition: opacity .18s ease, transform .18s ease;
      pointer-events: none;
    }
    .wd-toast.wd-in { opacity: 1; transform: translate(-50%, 0); }
    .wd-toast.wd-warn .wd-ic { color: #8A5A00; }
    .wd-ic { flex: none; color: #444746; }
    @media (prefers-reduced-motion: reduce) { .wd-toast { transition: none; } }
  `;

  let hostEl = null;
  let toastEl = null;
  let hideTimer = null;

  function build() {
    if (hostEl) return;
    hostEl = document.createElement("div");
    hostEl.id = HOST_ID;
    const shadow = hostEl.attachShadow({ mode: "open" });
    const style = document.createElement("style");
    style.textContent = CSS;
    shadow.appendChild(style);
    toastEl = document.createElement("div");
    toastEl.className = "wd-toast";
    toastEl.setAttribute("role", "status");
    toastEl.style.setProperty("--wd-font", WD.icons ? WD.icons.FONT : "Roboto, Arial, sans-serif");
    shadow.appendChild(toastEl);
    document.body.appendChild(hostEl);
  }

  /**
   * @param {string} text
   * @param {{tone?: "info"|"warn"}} [options]
   */
  function show(text, { tone = "info" } = {}) {
    try {
      build();
      const glyph = WD.icons ? WD.icons.icon(tone === "warn" ? "warning" : "info", 20) : "";
      toastEl.className = `wd-toast${tone === "warn" ? " wd-warn" : ""}`;
      toastEl.innerHTML = glyph;
      const label = document.createElement("span");
      label.textContent = text;
      toastEl.appendChild(label);
      requestAnimationFrame(() => toastEl && toastEl.classList.add("wd-in"));
      clearTimeout(hideTimer);
      hideTimer = setTimeout(() => toastEl && toastEl.classList.remove("wd-in"), VISIBLE_MS);
    } catch (err) {
      console.warn("wd: toast failed", err);
    }
  }

  function teardown() {
    clearTimeout(hideTimer);
    hideTimer = null;
    try {
      if (hostEl) hostEl.remove();
    } catch (err) {
      console.warn("wd: toast host remove failed", err);
    }
    hostEl = null;
    toastEl = null;
  }

  WD.toast = { show, teardown };
})();
