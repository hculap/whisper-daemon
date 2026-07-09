/**
 * rtc-tap.js — MAIN-world WebRTC audio tap for Google Meet.
 *
 * Runs at document_start in the page's MAIN world (the isolated content-script
 * world cannot see the page's RTCPeerConnection). It wraps RTCPeerConnection so
 * every remote audio track ('track' events fire for RECEIVER tracks only — the
 * local mic is a sender track and never arrives here, so it is excluded) is
 * observed. Capture stays INERT until the isolated orchestrator asks for it via
 * a window message; then it mixes all remote audio tracks through a GainNode
 * into an inlined AudioWorklet and posts 4096-sample Float32 PCM frames back to
 * the page for the orchestrator to relay as WD_PCM.
 *
 * No chrome.* APIs exist in the MAIN world — the ONLY channel is
 * window.postMessage. A chrome-extension:// worklet URL would also trip Meet's
 * CSP, so the worklet module is inlined as a Blob.
 *
 * Downstream is unchanged: page (WD_RTC_PCM) -> orchestrator (WD_PCM) -> SW ->
 * offscreen -> WebSocket -> daemon.
 */

"use strict";

(function () {
  const ORIGIN = location.origin;

  // Remote audio tracks currently present on any peer connection, keyed by id
  // (dedup). Lives for the page's lifetime; the RTC hook stays installed across
  // START/STOP so a later START immediately sees the current tracks.
  const remoteTracks = new Map(); // track.id -> MediaStreamTrack

  // Active-session graph. Rebuilt on each START, torn down on STOP.
  let capturing = false;
  let ctx = null;
  let gain = null;
  let workletNode = null;
  let workletUrl = null;
  const sourceNodes = new Map(); // track.id -> MediaStreamAudioSourceNode

  // The PCM worklet — same 4096-sample buffering as pcm-processor.js. Inlined
  // as a Blob module (chrome-extension:// URL is unavailable here and would hit
  // Meet's CSP anyway). No output pass-through: the node is never connected to
  // destination (avoids echoing the meeting locally).
  const WORKLET_CODE = `
class WdPcmProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._bufferSize = 4096;
    this._buffer = new Float32Array(this._bufferSize);
    this._bytesWritten = 0;
  }
  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;
    const channelData = input[0];
    for (let i = 0; i < channelData.length; i++) {
      this._buffer[this._bytesWritten++] = channelData[i];
      if (this._bytesWritten >= this._bufferSize) {
        this.port.postMessage(this._buffer);
        this._buffer = new Float32Array(this._bufferSize);
        this._bytesWritten = 0;
      }
    }
    return true;
  }
}
registerProcessor("wd-pcm", WdPcmProcessor);
`;

  function post(message) {
    try {
      window.postMessage(message, ORIGIN);
    } catch (err) {
      // Never throw into Meet's code.
      try { console.warn("wd-rtc: postMessage failed", err); } catch {}
    }
  }

  function postState() {
    post({ source: "wd-rtc", type: "WD_RTC_STATE", tracks: remoteTracks.size });
  }

  // --- Track bookkeeping (independent of whether we are capturing) ---

  function onTrackEvent(event) {
    try {
      const track = event && event.track;
      if (!track || track.kind !== "audio") return;
      if (remoteTracks.has(track.id)) return;
      remoteTracks.set(track.id, track);
      try {
        track.addEventListener("ended", () => removeTrack(track.id));
      } catch {}
      if (capturing) {
        wireTrack(track);
        postState();
      }
    } catch (err) {
      try { console.warn("wd-rtc: onTrackEvent failed", err); } catch {}
    }
  }

  function removeTrack(id) {
    try {
      remoteTracks.delete(id);
      const entry = sourceNodes.get(id);
      if (entry) {
        try { entry.node.disconnect(); } catch {}
        teardownSink(entry.sink);
        sourceNodes.delete(id);
      }
      if (capturing) postState();
    } catch {}
  }

  function teardownSink(sink) {
    try { if (sink) { sink.pause(); sink.srcObject = null; } } catch {}
  }

  // Route one remote audio track into the mixing GainNode. New tracks arriving
  // mid-recording (participants joining / Meet renegotiation) are wired here.
  function wireTrack(track) {
    try {
      if (!capturing || !ctx || !gain) return;
      if (sourceNodes.has(track.id)) return;
      // CRITICAL: a remote WebRTC track stays SILENT in WebAudio unless it is
      // ALSO attached to a PLAYING media element — Chrome only decodes a remote
      // track that has a real sink. A muted <audio> is that sink (muted = no
      // echo; the user already hears Meet's own playback). Empirically verified
      // in a live RTC loopback: source->worklet without this yields pure zeros;
      // with a muted playing <audio> the tap yields real PCM. Not connecting the
      // worklet to destination is fine — the <audio> element is the sink.
      const sink = new Audio();
      sink.muted = true;
      sink.srcObject = new MediaStream([track]);
      const played = sink.play();
      if (played && typeof played.catch === "function") played.catch(() => {});

      const node = ctx.createMediaStreamSource(new MediaStream([track]));
      node.connect(gain);
      sourceNodes.set(track.id, { node, sink });
    } catch (err) {
      try { console.warn("wd-rtc: wireTrack failed", err); } catch {}
    }
  }

  // --- Session lifecycle ---

  async function start() {
    try {
      if (capturing) return;
      ctx = new AudioContext({ sampleRate: 16000 });
      workletUrl = URL.createObjectURL(
        new Blob([WORKLET_CODE], { type: "application/javascript" })
      );
      await ctx.audioWorklet.addModule(workletUrl);

      gain = ctx.createGain();
      gain.gain.value = 1;
      workletNode = new AudioWorkletNode(ctx, "wd-pcm");
      // Mixer -> worklet only, NOT connected to destination (no echo). The remote
      // tracks are pulled by their per-track muted <audio> sinks (see wireTrack),
      // which is what actually makes their audio flow — the worklet then taps it.
      gain.connect(workletNode);
      workletNode.port.onmessage = (e) => {
        try {
          post({
            source: "wd-rtc",
            type: "WD_RTC_PCM",
            samples: Array.from(e.data),
          });
        } catch {}
      };

      // START is initiated from the record-button gesture, so autoplay allows
      // resume. If it stays suspended, the next START retries (fresh context).
      try { await ctx.resume(); } catch {}

      capturing = true;
      for (const track of remoteTracks.values()) wireTrack(track);
      postState();
    } catch (err) {
      try { console.warn("wd-rtc: start failed", err); } catch {}
      stop();
    }
  }

  function stop() {
    capturing = false;
    for (const entry of sourceNodes.values()) {
      try { entry.node.disconnect(); } catch {}
      teardownSink(entry.sink);
    }
    sourceNodes.clear();
    try { if (workletNode) workletNode.port.onmessage = null; } catch {}
    try { if (gain) gain.disconnect(); } catch {}
    try { if (workletNode) workletNode.disconnect(); } catch {}
    try { if (ctx) ctx.close(); } catch {}
    try { if (workletUrl) URL.revokeObjectURL(workletUrl); } catch {}
    ctx = null;
    gain = null;
    workletNode = null;
    workletUrl = null;
    // remoteTracks + the RTCPeerConnection hook stay in place for the next START.
  }

  // --- RTCPeerConnection hook (install once, at document_start) ---

  function installHook(name) {
    try {
      const Orig = window[name];
      if (typeof Orig !== "function") return;
      // A Proxy transparently preserves the prototype chain, static members
      // (generateCertificate, etc.), and instanceof — Meet keeps working; we
      // only add a 'track' listener without clobbering the page's ontrack.
      const proxied = new Proxy(Orig, {
        construct(target, args, newTarget) {
          const pc = Reflect.construct(target, args, newTarget);
          try {
            pc.addEventListener("track", onTrackEvent);
          } catch {}
          return pc;
        },
      });
      window[name] = proxied;
    } catch (err) {
      try { console.warn("wd-rtc: installHook failed", name, err); } catch {}
    }
  }

  installHook("RTCPeerConnection");
  installHook("webkitRTCPeerConnection");

  // --- Control channel (from the isolated orchestrator) ---

  window.addEventListener("message", (event) => {
    try {
      if (event.source !== window) return;
      const data = event.data;
      if (!data || data.source !== "wd") return;
      if (data.type === "WD_RTC_START") start();
      else if (data.type === "WD_RTC_STOP") stop();
    } catch (err) {
      try { console.warn("wd-rtc: message handler failed", err); } catch {}
    }
  });
})();
