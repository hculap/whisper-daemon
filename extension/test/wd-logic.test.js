"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");

const {
  captionsReducer,
  controlView,
  diffSettings,
  formatElapsed,
  isColdSetting,
  sourceView,
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

test("formatElapsed formats seconds as MM:SS, and H:MM:SS from one hour", () => {
  assert.equal(formatElapsed(0), "00:00");
  assert.equal(formatElapsed(5), "00:05");
  assert.equal(formatElapsed(65), "01:05");
  assert.equal(formatElapsed(600), "10:00");
  assert.equal(formatElapsed(3599), "59:59");
  assert.equal(formatElapsed(3661), "1:01:01");
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

test("sourceView reports a stale source as no signal with resting bars", () => {
  assert.deepEqual(sourceView({ rms: 0.3, age: 4 }), { status: "silent", label: "brak sygnału", bars: [4, 4, 4, 4, 4], level: 0 });
  assert.equal(sourceView(undefined).status, "silent");
});

test("sourceView separates speech from quiet at the daemon floor", () => {
  assert.equal(sourceView({ rms: 0.004, age: 0 }).label, "cisza");
  assert.equal(sourceView({ rms: 0.006, age: 0 }).label, "mowa");
  assert.deepEqual(sourceView({ rms: 0, age: 0 }).bars, [4, 4, 4, 4, 4]);
});

test("sourceView lights more bars for louder audio, all five near full scale", () => {
  const lit = (rms) => sourceView({ rms, age: 0 }).bars.filter((h) => h > 4).length;
  assert.ok(lit(0.006) < lit(0.05));
  assert.ok(lit(0.05) < lit(0.2));
  assert.equal(lit(0.5), 5);
  assert.equal(sourceView({ rms: 0.5, age: 0 }).level, 1);
});

test("controlView follows the daemon phase and shows the elapsed time", () => {
  assert.equal(controlView().mode, "idle");
  const rec = controlView({ daemonState: "recording", elapsed: 1458 });
  assert.equal(rec.mode, "recording");
  assert.equal(rec.time, "24:18");
  assert.match(rec.label, /Zatrzymaj nagrywanie, 24:18/);
  assert.equal(controlView({ daemonState: "transcribing" }).mode, "saving");
});

test("controlView puts pause and the participant warning over plain recording", () => {
  assert.equal(controlView({ daemonState: "recording", paused: true, warning: true }).mode, "paused");
  assert.equal(controlView({ daemonState: "recording", warning: true }).mode, "warning");
});

test("controlView shows connecting only before the daemon reports a phase", () => {
  assert.equal(controlView({ starting: true }).mode, "starting");
  assert.equal(controlView({ starting: true, daemonState: "recording" }).mode, "recording");
});

test("controlView tells an offline daemon from an unpaired extension", () => {
  assert.match(controlView({ connection: "offline" }).tooltip, /Daemon nie odpowiada/);
  assert.match(controlView({ connection: "unpaired" }).tooltip, /Sparuj/);
  assert.equal(controlView({ connection: "rejected" }).mode, "offline");
  assert.equal(controlView({ connection: "offline", daemonState: "recording" }).mode, "recording");
});

test("controlView names the shortcut it is given", () => {
  assert.match(controlView({ shortcut: "Ctrl+Shift+M" }).tooltip, /Ctrl\+Shift\+M/);
});
