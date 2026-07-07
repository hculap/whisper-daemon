/**
 * wd-logic.js — pure, DOM-free logic for the Meet integration.
 *
 * Loaded as a classic content script (shares the isolated-world scope with the
 * other wd-*.js files) AND requireable under node for unit tests. No `chrome`
 * usage, no DOM, no side effects — everything here is a pure function.
 */

"use strict";

// Cold settings only take effect on the NEXT recording; hot settings apply live.
const COLD_SETTINGS = Object.freeze([
  "recording_device",
  "recording_formats",
  "recording_dir",
  "diarize",
  "diarize_mode",
  "screenshot_displays",
  "capture_mic",
  "capture_tab",
]);

// Split on whitespace that follows a sentence terminator.
const SENTENCE_SPLIT = /(?<=[.!?…])\s+/;

// Hard cap on retained characters per sentence line. Whisper streaming output
// often lacks trailing punctuation, so an unpunctuated run would otherwise
// collapse into ONE sentence that grows without bound across chunks. Clamp each
// line to its trailing MAX_LINE_CHARS (on a word boundary) to bound the retained
// text — and therefore memory + DOM size — even for punctuation-free streams.
const MAX_LINE_CHARS = 240;

/**
 * Keep only the trailing `maxChars` of `line`, trimmed to a word boundary so a
 * clamp never splits a word mid-way.
 * @param {string} line
 * @param {number} maxChars
 * @returns {string}
 */
function clampLine(line, maxChars) {
  if (line.length <= maxChars) return line;
  const tail = line.slice(line.length - maxChars);
  const spaceIdx = tail.indexOf(" ");
  return spaceIdx > 0 ? tail.slice(spaceIdx + 1) : tail;
}

/**
 * Reduce accumulated caption lines with newly transcribed text.
 *
 * Combines the existing lines with `newText`, re-splits into sentences, trims
 * empties, and keeps only the last `maxSentences`. Each retained sentence is
 * clamped to `maxCharsPerLine` so a punctuation-free stream cannot grow without
 * limit. Deterministic and immutable.
 *
 * @param {string[]} lines - current caption lines
 * @param {string} newText - freshly transcribed text
 * @param {number} [maxSentences=2] - how many trailing sentences to keep
 * @param {number} [maxCharsPerLine=MAX_LINE_CHARS] - per-line character cap
 * @returns {string[]} a new array of caption lines
 */
function captionsReducer(
  lines,
  newText,
  maxSentences = 2,
  maxCharsPerLine = MAX_LINE_CHARS
) {
  const safeLines = Array.isArray(lines) ? lines : [];
  const addition = typeof newText === "string" ? newText.trim() : "";

  const combined = [...safeLines, addition]
    .map((part) => (typeof part === "string" ? part.trim() : ""))
    .filter((part) => part.length > 0)
    .join(" ");

  if (combined.length === 0) return [];

  const sentences = combined
    .split(SENTENCE_SPLIT)
    .map((s) => s.trim())
    .filter((s) => s.length > 0);

  const cap = typeof maxCharsPerLine === "number" && maxCharsPerLine > 0
    ? maxCharsPerLine
    : MAX_LINE_CHARS;

  return sentences
    .slice(-Math.max(1, maxSentences))
    .map((s) => clampLine(s, cap));
}

/**
 * Compute the minimal patch of keys whose value differs between `current` and
 * `next`. Only keys present in `next` are considered. Deep-compares by value.
 *
 * @param {object} current
 * @param {object} next
 * @returns {object} keys from `next` whose value changed
 */
function diffSettings(current, next) {
  const base = current && typeof current === "object" ? current : {};
  const target = next && typeof next === "object" ? next : {};

  return Object.keys(target).reduce((patch, key) => {
    const equal = JSON.stringify(base[key]) === JSON.stringify(target[key]);
    return equal ? patch : { ...patch, [key]: target[key] };
  }, {});
}

/**
 * Format elapsed seconds as "MM:SS". Invalid/negative input clamps to "00:00".
 *
 * @param {number} seconds
 * @returns {string}
 */
function formatElapsed(seconds) {
  const total =
    typeof seconds === "number" && Number.isFinite(seconds) && seconds > 0
      ? Math.floor(seconds)
      : 0;
  const mm = String(Math.floor(total / 60)).padStart(2, "0");
  const ss = String(total % 60).padStart(2, "0");
  return `${mm}:${ss}`;
}

/**
 * @param {string} key
 * @returns {boolean} true if the setting only applies to the next recording
 */
function isColdSetting(key) {
  return COLD_SETTINGS.includes(key);
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    COLD_SETTINGS,
    captionsReducer,
    diffSettings,
    formatElapsed,
    isColdSetting,
  };
} else if (typeof window !== "undefined") {
  // Content-script context: publish on the shared WD namespace so the other
  // wd-*.js files can reach these helpers without relying on hoisting order.
  window.WD = window.WD || {};
  window.WD.logic = {
    COLD_SETTINGS,
    captionsReducer,
    diffSettings,
    formatElapsed,
    isColdSetting,
  };
}
