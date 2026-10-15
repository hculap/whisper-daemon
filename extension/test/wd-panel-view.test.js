const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

// The menu markup builders run as classic scripts on window.WD, as in Meet.
function loadView() {
  const context = vm.createContext({});
  context.window = context;
  for (const file of ['wd-logic.js', 'wd-icons.js', 'wd-panel-view.js']) {
    vm.runInContext(fs.readFileSync(path.join(__dirname, '..', file), 'utf8'), context, {filename: file});
  }
  return context.WD.panelView;
}

const view = loadView();
const SETTINGS = {capture_mic: true, capture_tab: false, recording_device: '', recording_language: 'pl',
  recording_formats: ['txt'], recording_dir: '~/Nagrania', diarize: true, live_captions: true,
  capture_screenshots: false, screenshot_displays: 'primary'};
const BASE = {settings: SETTINGS, devices: ['USB <Mic>'], daemonState: 'idle', starting: false, paused: false,
  elapsed: 3725, health: null, warning: null, connection: 'connected', title: 'Plan <Q4>',
  diarization: null, subview: 'main', shortcut: '⌘⇧M'};
const render = overrides => view.render({...BASE, ...overrides});

test('every menu state renders without unfilled values', () => {
  const states = [{}, {daemonState: 'recording'}, {daemonState: 'recording', paused: true},
    {daemonState: 'recording', subview: 'settings'}, {daemonState: 'transcribing'}, {starting: true},
    {connection: 'offline', settings: null}, {connection: 'unpaired', settings: null},
    {daemonState: 'recording', warning: {id: 'silence', title: 'Cisza', body: 'Sprawdź głośnik.'}}];
  for (const state of states) {
    const html = render(state);
    assert.doesNotMatch(html, /undefined|NaN|\[object Object\]/, JSON.stringify(state));
  }
});

test('the idle menu offers start and shows the saved settings', () => {
  const html = render({});
  assert.match(html, /data-action="toggle"[^>]*>[\s\S]*Rozpocznij nagrywanie/);
  assert.match(html, /data-switch="capture_tab"[^>]*aria-checked="false"|aria-checked="false"[^>]*data-switch="capture_tab"/);
  assert.match(html, /value="pl" selected/);
  assert.match(html, /data-disp="primary" aria-pressed="true"/);
  assert.match(html, /value="~\/Nagrania"/);
});

test('the recording menu shows the timer, pause and stop, and hides cold settings behind a row', () => {
  const html = render({daemonState: 'recording'});
  assert.match(html, /data-el="timer"[^>]*>1:02:05</);
  assert.match(html, /data-action="pause"/);
  assert.match(html, /Zakończ i zapisz/);
  assert.match(html, /data-action="settings"/);
  assert.doesNotMatch(html, /data-key="recording_language"/);
  assert.match(render({daemonState: 'recording', paused: true}), /Wznów/);
});

test('a participant warning carries its action and an ignore button', () => {
  const html = render({daemonState: 'recording',
    warning: {id: 'no-tracks', title: 'Brak', body: 'Udostępnij kartę.', action: 'Udostępnij kartę'}});
  assert.match(html, /data-action="warning-action"[^>]*>Udostępnij kartę</);
  assert.match(html, /data-action="warning-dismiss"/);
});

test('an unpaired extension is offered pairing instead of a start that cannot work', () => {
  const html = render({connection: 'unpaired', settings: null});
  assert.match(html, /data-action="pair"[^>]*>[\s\S]*Sparuj rozszerzenie/);
  assert.doesNotMatch(html, /Rozpocznij nagrywanie/);
  assert.match(render({connection: 'offline', settings: null}), /data-action="reconnect"/);
});

test('names from the daemon and the page are escaped', () => {
  const html = render({});
  assert.match(html, /USB &lt;Mic&gt;/);
  assert.match(html, /Plan &lt;Q4&gt;/);
  assert.doesNotMatch(html, /<Mic>|<Q4>/);
});
