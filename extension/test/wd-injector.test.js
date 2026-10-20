"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const BOTTOM = { left: 900, top: 1090, width: 48, height: 48 };
const HEADER = { left: 20, top: 8, width: 48, height: 48 };

function fixture(slotRect, {automaticMutations = false} = {}) {
  let mutationCallback = null;
  let observerConnected = false;
  const mutationRecords = [];
  const animationFrames = [];

  function queueMutation(record) {
    if (automaticMutations && observerConnected) mutationRecords.push(record);
  }

  class ClassList {
    constructor(element) { this.element = element; }
    contains(name) { return this.element.className.split(/\s+/).includes(name); }
    add(name) { if (!this.contains(name)) this.element.className += ` ${name}`; }
    remove(name) { this.element.className = this.element.className.split(/\s+/).filter(x => x !== name).join(" "); }
  }
  class Element {
    constructor(tag, rect = null) {
      this.tagName = tag.toUpperCase(); this.rect = rect; this.children = [];
      this.parentElement = null; this.id = ""; this.className = ""; this.dataset = {};
      this.style = { setProperty() {} }; this.classList = new ClassList(this);
    }
    appendChild(child) { return this.insertBefore(child, null); }
    insertBefore(child, before) {
      if (child.parentElement) child.remove();
      const index = before ? this.children.indexOf(before) : -1;
      if (index < 0) this.children.push(child); else this.children.splice(index, 0, child);
      child.parentElement = this;
      queueMutation({target: this, addedNodes: [child], removedNodes: []});
      return child;
    }
    remove() {
      if (!this.parentElement) return;
      const parent = this.parentElement;
      const index = parent.children.indexOf(this);
      if (index >= 0) parent.children.splice(index, 1);
      this.parentElement = null;
      queueMutation({target: parent, addedNodes: [], removedNodes: [this]});
    }
    contains(other) { return this === other || this.children.some(child => child.contains(other)); }
    get nextElementSibling() {
      if (!this.parentElement) return null;
      return this.parentElement.children[this.parentElement.children.indexOf(this) + 1] || null;
    }
    getBoundingClientRect() { return this.rect || this.parentElement?.getBoundingClientRect() || {left:0,top:0,width:0,height:0}; }
    setAttribute() {}
    addEventListener() {}
    querySelector(selector) {
      const wanted = selector.startsWith(".") ? selector.slice(1) : null;
      if (wanted && this.classList.contains(wanted)) return this;
      for (const child of this.children) { const found = child.querySelector(selector); if (found) return found; }
      return null;
    }
    querySelectorAll() { return []; }
    // Like the DOM, rewriting markup or text replaces child nodes: a childList
    // mutation that the body observer sees while the element is attached.
    set innerHTML(value) {
      this.children = [];
      if (String(value).includes('class="wd-time"')) {
        const time = new Element("span");
        time.className = "wd-time";
        time.parentElement = this;
        this.children.push(time);
      }
      if (body.contains(this)) queueMutation({target: this, addedNodes: [], removedNodes: []});
    }
    get textContent() { return this.text || ""; }
    set textContent(value) {
      this.text = String(value);
      if (body.contains(this)) queueMutation({target: this, addedNodes: [], removedNodes: []});
    }
  }

  const html = new Element("html");
  const head = new Element("head");
  const body = new Element("body", {left:0,top:0,width:2048,height:1152});
  html.appendChild(head); html.appendChild(body);
  const slot = new Element("div", slotRect); slot.id = "browser-extension-end-buttons"; body.appendChild(slot);
  let present = [];
  let cameras = [];
  const walk = (root, predicate) => {
    if (predicate(root)) return root;
    for (const child of root.children) { const found = walk(child, predicate); if (found) return found; }
    return null;
  };
  const document = {
    head, body, documentElement: html,
    createElement: tag => new Element(tag),
    getElementById: id => walk(html, node => node.id === id),
    querySelectorAll: selector => selector.includes("Present") ? present : cameras,
    querySelector: () => null,
  };
  class MutationObserver {
    constructor(callback) { mutationCallback = callback; }
    observe() { observerConnected = true; }
    disconnect() { observerConnected = false; mutationRecords.length = 0; }
    takeRecords() { return mutationRecords.splice(0); }
  }
  const context = vm.createContext({console, document, MutationObserver,
    requestAnimationFrame: callback => {
      if (automaticMutations) animationFrames.push(callback); else callback();
    }, innerWidth:2048, innerHeight:1152});
  context.window = context;
  for (const script of ["wd-logic.js", "wd-injector.js"]) {
    vm.runInContext(fs.readFileSync(path.join(__dirname, "..", script), "utf8"), context, {filename:script});
  }
  return {
    injector: context.WD.injector,
    document,
    slot,
    mutate() {
      const record = {target: body, addedNodes: [], removedNodes: []};
      if (automaticMutations) mutationRecords.push(record); else mutationCallback([record]);
    },
    flush(maxSteps = 20) {
      let steps = 0;
      while ((mutationRecords.length || animationFrames.length) && steps < maxSteps) {
        steps += 1;
        if (mutationRecords.length) {
          const records = mutationRecords.splice(0);
          mutationCallback(records);
        } else {
          animationFrames.shift()();
        }
      }
      return {drained: mutationRecords.length === 0 && animationFrames.length === 0, steps};
    },
    moveSlot(rect) { slot.rect = rect; },
    mountToolbar() {
      const row = new Element("div", BOTTOM);
      const cameraCell = new Element("div", BOTTOM);
      const presentCell = new Element("div", {...BOTTOM, left:1050});
      const camera = new Element("button", BOTTOM);
      const share = new Element("button", {...BOTTOM, left:1050});
      cameraCell.appendChild(camera); presentCell.appendChild(share);
      row.appendChild(cameraCell); row.appendChild(presentCell); body.appendChild(row);
      cameras = [camera]; present = [share];
      return {row, presentCell};
    },
  };
}

test("a header fallback is rejected", () => {
  const f = fixture(HEADER);
  assert.equal(f.injector.injectControl(), null);
  assert.equal(f.document.getElementById("wd-control"), null);
});

test("a bottom fallback is promoted when the preferred toolbar row appears", () => {
  const f = fixture(BOTTOM);
  const control = f.injector.injectControl();
  assert.equal(control.parentElement, f.slot);
  f.injector.startObserver(() => f.injector.injectControl());
  const {row, presentCell} = f.mountToolbar();
  f.mutate();
  assert.equal(control.parentElement, row);
  assert.equal(control.nextElementSibling, presentCell);
});

test("a fallback moved into the header is removed until a safe row exists", () => {
  const f = fixture(BOTTOM);
  const oldControl = f.injector.injectControl();
  f.injector.startObserver(() => f.injector.injectControl());
  f.moveSlot(HEADER);
  f.mutate();
  assert.equal(oldControl.parentElement, null);
  assert.equal(f.document.getElementById("wd-control"), null);
  const {row} = f.mountToolbar();
  f.mutate();
  assert.equal(f.document.getElementById("wd-control").parentElement, row);
});

test("rejecting a header fallback does not trigger an observer retry loop", () => {
  const f = fixture(HEADER, {automaticMutations: true});
  let attempts = 0;
  f.injector.startObserver(() => {
    attempts += 1;
    f.injector.injectControl();
  });

  f.mutate();
  const result = f.flush();

  assert.equal(result.drained, true, `observer still active after ${result.steps} steps`);
  assert.equal(attempts, 1);
  assert.equal(f.document.getElementById("wd-control"), null);
});

test("an unchanged timer repaint does not wake the observer", () => {
  const f = fixture(BOTTOM, {automaticMutations: true});
  const control = f.injector.injectControl();
  assert.equal(control.parentElement, f.slot);
  let attempts = 0;
  f.injector.startObserver(() => {
    attempts += 1;
    f.injector.injectControl();
  });
  const time = () => control.querySelector(".wd-time").textContent;

  f.injector.setView({mode: "recording", time: "0:05", label: "Stop"});
  f.flush();
  assert.equal(time(), "0:05");
  const afterFirstPaint = attempts;

  // The 250 ms tick repaints three times per displayed second.
  for (let i = 0; i < 3; i += 1) f.injector.setView({time: "0:05"});
  const idle = f.flush();
  assert.equal(idle.steps, 0);
  assert.equal(attempts, afterFirstPaint);

  f.injector.setView({time: "0:06"});
  f.flush();
  assert.equal(time(), "0:06");
  assert.equal(attempts, afterFirstPaint + 1);
});
