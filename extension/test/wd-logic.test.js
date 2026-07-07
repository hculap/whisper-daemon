"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");

const {
  captionsReducer,
  diffSettings,
  formatElapsed,
  isColdSetting,
} = require("../wd-logic.js");

test("captionsReducer appends first text as a single line", () => {
  const out = captionsReducer([], "Hello world.");
  assert.deepEqual(out, ["Hello world."]);
});

test("captionsReducer splits combined text into sentences", () => {
  const out = captionsReducer(["Hello world."], "How are you? I am fine.");
  assert.deepEqual(out, ["How are you?", "I am fine."]);
});

test("captionsReducer keeps only the last maxSentences (default 2)", () => {
  const out = captionsReducer([], "One. Two. Three. Four.");
  assert.deepEqual(out, ["Three.", "Four."]);
});

test("captionsReducer honors a custom maxSentences", () => {
  const out = captionsReducer([], "One. Two. Three. Four.", 3);
  assert.deepEqual(out, ["Two.", "Three.", "Four."]);
});

test("captionsReducer handles all terminal punctuation …/./!/?", () => {
  const out = captionsReducer([], "Wait… Really! Yes? Ok.", 4);
  assert.deepEqual(out, ["Wait…", "Really!", "Yes?", "Ok."]);
});

test("captionsReducer trims empty sentences and whitespace", () => {
  const out = captionsReducer([], "  Alpha.   Beta.   ");
  assert.deepEqual(out, ["Alpha.", "Beta."]);
});

test("captionsReducer keeps a trailing unterminated fragment", () => {
  const out = captionsReducer(["Done."], "and still typing");
  assert.deepEqual(out, ["Done.", "and still typing"]);
});

test("captionsReducer accumulates across calls", () => {
  let lines = [];
  lines = captionsReducer(lines, "First sentence.");
  lines = captionsReducer(lines, "Second sentence.");
  lines = captionsReducer(lines, "Third sentence.");
  assert.deepEqual(lines, ["Second sentence.", "Third sentence."]);
});

test("captionsReducer ignores empty newText", () => {
  const out = captionsReducer(["Kept."], "   ");
  assert.deepEqual(out, ["Kept."]);
});

test("captionsReducer does not mutate the input array", () => {
  const input = ["A."];
  const frozen = Object.freeze(input.slice());
  const out = captionsReducer(frozen, "B.");
  assert.deepEqual(input, ["A."]);
  assert.notEqual(out, input);
});

test("formatElapsed formats seconds as MM:SS", () => {
  assert.equal(formatElapsed(0), "00:00");
  assert.equal(formatElapsed(5), "00:05");
  assert.equal(formatElapsed(65), "01:05");
  assert.equal(formatElapsed(600), "10:00");
  assert.equal(formatElapsed(3661), "61:01");
});

test("formatElapsed floors fractional seconds", () => {
  assert.equal(formatElapsed(9.9), "00:09");
});

test("formatElapsed clamps negative and invalid to 00:00", () => {
  assert.equal(formatElapsed(-5), "00:00");
  assert.equal(formatElapsed(NaN), "00:00");
  assert.equal(formatElapsed(undefined), "00:00");
});

test("isColdSetting marks cold keys", () => {
  for (const key of [
    "recording_device",
    "recording_formats",
    "recording_dir",
    "diarize",
    "diarize_mode",
    "screenshot_displays",
    "capture_mic",
    "capture_tab",
  ]) {
    assert.equal(isColdSetting(key), true, `${key} should be cold`);
  }
});

test("isColdSetting marks hot keys", () => {
  assert.equal(isColdSetting("live_captions"), false);
  assert.equal(isColdSetting("capture_screenshots"), false);
});

test("isColdSetting is false for unknown keys", () => {
  assert.equal(isColdSetting("nonexistent"), false);
});

test("diffSettings returns only changed keys with next values", () => {
  const current = { capture_mic: true, recording_dir: "~/Desktop", diarize: false };
  const next = { capture_mic: false, recording_dir: "~/Desktop", diarize: true };
  assert.deepEqual(diffSettings(current, next), { capture_mic: false, diarize: true });
});

test("diffSettings returns empty object when nothing changed", () => {
  const current = { a: 1, b: "x" };
  const next = { a: 1, b: "x" };
  assert.deepEqual(diffSettings(current, next), {});
});

test("diffSettings deep-compares array values", () => {
  const current = { recording_formats: ["txt"] };
  const nextSame = { recording_formats: ["txt"] };
  const nextDiff = { recording_formats: ["txt", "srt"] };
  assert.deepEqual(diffSettings(current, nextSame), {});
  assert.deepEqual(diffSettings(current, nextDiff), { recording_formats: ["txt", "srt"] });
});

test("diffSettings only considers keys present in next", () => {
  const current = { a: 1, b: 2 };
  const next = { a: 2 };
  assert.deepEqual(diffSettings(current, next), { a: 2 });
});

test("diffSettings treats missing current key as changed", () => {
  const current = {};
  const next = { live_captions: true };
  assert.deepEqual(diffSettings(current, next), { live_captions: true });
});
