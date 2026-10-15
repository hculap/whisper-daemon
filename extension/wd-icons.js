/**
 * wd-icons.js — inline stroke icons shared by the Meet surfaces (bar control,
 * menu, toast) and the popup. Kept inline so nothing loads over the network.
 *
 * Classic script: a content script in Meet, a plain script in the popup.
 * Exposes WD.icons.
 */

"use strict";

(function () {
  const WD = (window.WD = window.WD || {});

  // 24×24 paths drawn with currentColor strokes; filled parts set their own fill.
  const PATHS = {
    rec: '<circle cx="8.5" cy="12" r="3.5" fill="currentColor" stroke="none"/><path d="M14 8.5a5 5 0 0 1 0 7"/><path d="M17.2 5.8a9 9 0 0 1 0 12.4"/>',
    recOff: '<circle cx="8.5" cy="12" r="3.5" fill="currentColor" stroke="none"/><path d="M14 8.5a5 5 0 0 1 0 7"/><path d="M17.2 5.8a9 9 0 0 1 0 12.4"/><path d="M4 4l16 16"/>',
    chevUp: '<path d="M7 14.5 12 9.5l5 5"/>',
    chevDown: '<path d="M7 9.5 12 14.5l5-5"/>',
    chevRight: '<path d="M10 7l5 5-5 5"/>',
    back: '<path d="M19 12H5"/><path d="M11 6l-6 6 6 6"/>',
    pause: '<path d="M9 6.5v11"/><path d="M15 6.5v11"/>',
    play: '<path d="M8 5.5v13l10.5-6.5z" fill="currentColor"/>',
    stop: '<rect x="5" y="5" width="14" height="14" rx="3" fill="currentColor" stroke="none"/>',
    spinner: '<circle cx="12" cy="12" r="8" stroke-opacity=".3"/><path d="M12 4a8 8 0 0 1 8 8"/>',
    me: '<circle cx="12" cy="8" r="3.4"/><path d="M5 20a7 7 0 0 1 14 0"/>',
    people: '<circle cx="9" cy="8.5" r="3"/><path d="M3.2 19a5.8 5.8 0 0 1 11.6 0"/><path d="M15.5 6.2a3 3 0 0 1 0 5.6"/><path d="M17 13.4a5.8 5.8 0 0 1 3.8 5.6"/>',
    mic: '<rect x="9" y="3" width="6" height="11" rx="3"/><path d="M6 11a6 6 0 0 0 12 0"/><path d="M12 17v4"/>',
    lang: '<circle cx="12" cy="12" r="9"/><path d="M3 12h18"/><path d="M12 3c2.5 2.4 3.8 5.6 3.8 9s-1.3 6.6-3.8 9c-2.5-2.4-3.8-5.6-3.8-9S9.5 5.4 12 3z"/>',
    diar: '<circle cx="9" cy="9" r="3.2"/><path d="M3.5 20a5.5 5.5 0 0 1 11 0"/><path d="M16 8a3.6 3.6 0 0 1 0 8"/><path d="M18.4 6a6.8 6.8 0 0 1 0 12"/>',
    shot: '<rect x="3" y="6" width="18" height="13" rx="2.5"/><circle cx="12" cy="12.5" r="3.4"/>',
    cc: '<rect x="3" y="5" width="18" height="14" rx="2.5"/><path d="M7.5 13h3"/><path d="M13.5 13h3.5"/>',
    folder: '<path d="M3.5 7a2 2 0 0 1 2-2h3.2l2 2H18.5a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-13a2 2 0 0 1-2-2z"/>',
    doc: '<path d="M7 3.5h7l4 4V19a1.5 1.5 0 0 1-1.5 1.5h-9A1.5 1.5 0 0 1 6 19V5A1.5 1.5 0 0 1 7 3.5z"/><path d="M13.5 3.5V8H18"/><path d="M9 13h6"/><path d="M9 16h4"/>',
    sliders: '<path d="M4 7h10"/><path d="M18 7h2"/><path d="M4 17h4"/><path d="M12 17h8"/><circle cx="16" cy="7" r="2"/><circle cx="10" cy="17" r="2"/>',
    check: '<path d="M5 12.5l4.5 4.5L19 7.5"/>',
    warning: '<path d="M12 4 21 19.5H3z"/><path d="M12 10v4"/><path d="M12 16.8h.01"/>',
    pencil: '<path d="M5 19l1-4L15.5 5.5a2.1 2.1 0 0 1 3 3L9 18z"/>',
    info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v5"/><path d="M12 8h.01"/>',
    camera: '<rect x="3" y="6.5" width="12.5" height="11" rx="2.5"/><path d="M15.5 10.5 21 7.5v9l-5.5-3z"/>',
    tab: '<rect x="3" y="5" width="18" height="14" rx="2.5"/><path d="M3 9.5h18"/>',
    copy: '<rect x="8.5" y="8.5" width="11" height="11" rx="2"/><path d="M15.5 8.5V6a1.5 1.5 0 0 0-1.5-1.5H6A1.5 1.5 0 0 0 4.5 6v8A1.5 1.5 0 0 0 6 15.5h2.5"/>',
    download: '<path d="M12 4v11"/><path d="M7.5 10.5 12 15l4.5-4.5"/><path d="M5 19.5h14"/>',
    trash: '<path d="M5 7h14"/><path d="M10 7V5h4v2"/><path d="M7 7l1 12.5h8L17 7"/>',
    lock: '<rect x="5" y="10.5" width="14" height="10" rx="2"/><path d="M8 10.5V8a4 4 0 0 1 8 0v2.5"/>',
  };

  /**
   * @param {string} name
   * @param {number} [size]
   * @param {{stroke?: number, cls?: string}} [options]
   * @returns {string} SVG markup
   */
  function icon(name, size = 22, { stroke = 1.8, cls = "" } = {}) {
    return `<svg class="wd-ic ${cls}" width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" ` +
      `stroke="currentColor" stroke-width="${stroke}" stroke-linecap="round" stroke-linejoin="round" ` +
      `aria-hidden="true">${PATHS[name] || ""}</svg>`;
  }

  // Meet loads Google Sans; inheriting it keeps every surface native.
  const FONT = '"Google Sans", "Google Sans Text", Roboto, Arial, sans-serif';

  WD.icons = { icon, FONT };
})();
