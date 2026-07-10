/**
 * wd-captions.js — film-style live caption band in its OWN Shadow DOM host.
 * Fixed bottom-center, above Meet's native CC, semi-transparent dark bg, max
 * two lines, new line fades in / old fades out.
 *
 * Classic content script sharing the isolated-world scope. Exposes WD.captions.
 */

"use strict";

(function () {
  const WD = (window.WD = window.WD || {});

  const HOST_ID = "wd-captions-host";

  const CSS = `
    :host { all: initial; }
    .wd-band {
      position: fixed;
      bottom: 15%;
      left: 50%;
      transform: translateX(-50%);
      z-index: 2147483645;
      max-width: 72vw;
      display: none;
      flex-direction: column;
      align-items: center;
      gap: 4px;
      pointer-events: none;
      font-family: "Google Sans", Roboto, Arial, sans-serif;
    }
    .wd-band.wd-visible { display: flex; }
    .wd-line {
      background: rgba(0, 0, 0, 0.72);
      color: #fff;
      padding: 4px 14px;
      border-radius: 6px;
      font-size: 22px;
      line-height: 1.35;
      text-align: center;
      text-shadow: 0 1px 2px rgba(0,0,0,0.6);
      opacity: 0;
      transition: opacity 0.35s ease;
      max-width: 100%;
    }
    .wd-line.wd-in { opacity: 1; }
  `;

  let hostEl = null;
  let shadow = null;
  let bandEl = null;
  let visible = false;
  let lastLines = [];

  function build() {
    if (hostEl) return;
    hostEl = document.createElement("div");
    hostEl.id = HOST_ID;
    shadow = hostEl.attachShadow({ mode: "open" });

    const style = document.createElement("style");
    style.textContent = CSS;
    shadow.appendChild(style);

    bandEl = document.createElement("div");
    bandEl.className = "wd-band";
    shadow.appendChild(bandEl);

    document.body.appendChild(hostEl);
  }

  function apply() {
    if (!bandEl) return;
    bandEl.classList.toggle("wd-visible", visible && lastLines.length > 0);
  }

  /**
   * Render caption lines (already reduced by captionsReducer).
   * @param {string[]} lines
   */
  function render(lines) {
    build();
    lastLines = Array.isArray(lines) ? lines.filter((l) => l && l.trim()) : [];

    // Rebuild lines; each starts hidden then fades in on next frame.
    bandEl.textContent = "";
    for (const text of lastLines) {
      const div = document.createElement("div");
      div.className = "wd-line";
      div.textContent = text;
      bandEl.appendChild(div);
    }
    requestAnimationFrame(() => {
      // teardown() (on MEET_LEFT) can null bandEl before this frame runs.
      if (bandEl) bandEl.querySelectorAll(".wd-line").forEach((el) => el.classList.add("wd-in"));
    });
    apply();
  }

  function setVisible(on) {
    build();
    visible = !!on;
    if (!visible) lastLines = [];
    if (!visible && bandEl) bandEl.textContent = "";
    apply();
  }

  function clear() {
    lastLines = [];
    if (bandEl) bandEl.textContent = "";
    apply();
  }

  /**
   * Remove the caption shadow host from the document. Safe to call repeatedly;
   * a later render()/setVisible() rebuilds it on demand.
   */
  function teardown() {
    try {
      if (hostEl) hostEl.remove();
    } catch (err) {
      console.warn("wd: captions host remove failed", err);
    }
    hostEl = null;
    shadow = null;
    bandEl = null;
    visible = false;
    lastLines = [];
  }

  WD.captions = {
    render,
    setVisible,
    clear,
    teardown,
  };
})();
