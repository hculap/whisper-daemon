/**
 * Floating overlay widget on Google Meet pages.
 * Shows recording status, timer, live transcript, and stop button.
 * Recording is started via the popup (Chrome tabCapture requires user gesture).
 */

(function () {
  if (document.getElementById("wd-overlay")) return;

  const STYLES = `
    #wd-overlay {
      position: fixed;
      top: 16px;
      right: 16px;
      z-index: 999999;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 13px;
      color: #222;
      user-select: none;
    }
    #wd-pill {
      display: flex;
      align-items: center;
      gap: 8px;
      background: rgba(255, 255, 255, 0.95);
      backdrop-filter: blur(8px);
      border-radius: 20px;
      padding: 6px 14px;
      box-shadow: 0 2px 12px rgba(0,0,0,0.15);
      cursor: grab;
    }
    #wd-pill:hover { box-shadow: 0 4px 20px rgba(0,0,0,0.2); }
    #wd-pill.dragging { cursor: grabbing; }
    #wd-dot {
      width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0;
      background: #9E9E9E;
    }
    #wd-dot.capturing {
      background: #F44336;
      animation: wd-pulse 1.5s infinite;
    }
    #wd-dot.starting { background: #FF9800; }
    #wd-dot.detected { background: #FF9800; }
    @keyframes wd-pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.4; }
    }
    #wd-label {
      font-size: 12px; font-weight: 500; color: #555;
      max-width: 120px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    }
    #wd-timer {
      font-size: 13px; font-weight: 600; color: #F44336;
      display: none; font-variant-numeric: tabular-nums;
    }
    #wd-timer.visible { display: inline; }
    #wd-btn {
      border: none; border-radius: 14px; padding: 4px 12px;
      font-size: 11px; font-weight: 600; cursor: pointer;
      background: #424242; color: white; display: none;
    }
    #wd-btn:hover { background: #212121; }
    #wd-btn.visible { display: inline-block; }
    #wd-expand {
      border: none; background: none; cursor: pointer;
      font-size: 14px; color: #888; padding: 0 2px; line-height: 1;
    }
    #wd-transcript {
      display: none;
      background: rgba(255, 255, 255, 0.95);
      backdrop-filter: blur(8px);
      border-radius: 12px; margin-top: 6px; padding: 10px 14px;
      box-shadow: 0 2px 12px rgba(0,0,0,0.15);
      max-height: 200px; overflow-y: auto;
      font-size: 12px; color: #444; line-height: 1.5;
    }
    #wd-transcript.visible { display: block; }
    .wd-chunk { padding: 3px 0; border-bottom: 1px solid #EEE; }
    .wd-chunk:last-child { border-bottom: none; }
  `;

  const style = document.createElement("style");
  style.textContent = STYLES;
  document.head.appendChild(style);

  const overlay = document.createElement("div");
  overlay.id = "wd-overlay";
  overlay.innerHTML = `
    <div id="wd-pill">
      <span id="wd-dot"></span>
      <span id="wd-label">Whisper</span>
      <span id="wd-timer">00:00</span>
      <button id="wd-btn">Stop</button>
      <button id="wd-expand">▼</button>
    </div>
    <div id="wd-transcript"></div>
  `;
  document.body.appendChild(overlay);

  const dot = document.getElementById("wd-dot");
  const label = document.getElementById("wd-label");
  const timerEl = document.getElementById("wd-timer");
  const btn = document.getElementById("wd-btn");
  const expandBtn = document.getElementById("wd-expand");
  const transcriptEl = document.getElementById("wd-transcript");
  const pill = document.getElementById("wd-pill");

  let currentState = "idle";
  let timerInterval = null;
  let recordingStart = null;
  let transcriptExpanded = false;

  // --- State ---

  function syncState() {
    chrome.runtime.sendMessage({ type: "GET_STATE" }, (res) => {
      if (res) updateUI(res.state, res.meetTitle);
    });
  }

  function updateUI(state, title) {
    currentState = state;
    dot.className = "";
    dot.id = "wd-dot";
    if (state !== "idle") dot.classList.add(state);

    if (state === "capturing") {
      btn.classList.add("visible");
      startTimer();
    } else {
      btn.classList.remove("visible");
      stopTimer();
    }

    label.textContent = title || "Whisper";
  }

  // --- Messages ---

  chrome.runtime.onMessage.addListener((msg) => {
    if (msg.type === "CAPTURE_STARTED") {
      updateUI("capturing");
    } else if (msg.type === "CAPTURE_STOPPED" || msg.type === "CAPTURE_ERROR") {
      updateUI("idle");
    } else if (msg.type === "CHUNK_TRANSCRIBED") {
      addChunk(msg.text);
    }
  });

  // --- Stop button ---

  btn.addEventListener("click", (e) => {
    e.stopPropagation();
    if (currentState === "capturing") {
      chrome.runtime.sendMessage({ type: "STOP_RECORDING" });
    }
  });

  // --- Expand transcript ---

  expandBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    transcriptExpanded = !transcriptExpanded;
    transcriptEl.classList.toggle("visible", transcriptExpanded);
    expandBtn.textContent = transcriptExpanded ? "▲" : "▼";
  });

  // --- Timer ---

  function startTimer() {
    if (timerInterval) return;
    recordingStart = Date.now();
    timerEl.classList.add("visible");
    updateTimerDisplay();
    timerInterval = setInterval(updateTimerDisplay, 1000);
  }

  function stopTimer() {
    clearInterval(timerInterval);
    timerInterval = null;
    timerEl.classList.remove("visible");
  }

  function updateTimerDisplay() {
    if (!recordingStart) return;
    const elapsed = Math.floor((Date.now() - recordingStart) / 1000);
    const m = String(Math.floor(elapsed / 60)).padStart(2, "0");
    const s = String(elapsed % 60).padStart(2, "0");
    timerEl.textContent = `${m}:${s}`;
  }

  // --- Transcript ---

  function addChunk(text) {
    const div = document.createElement("div");
    div.className = "wd-chunk";
    div.textContent = text;
    transcriptEl.appendChild(div);
    transcriptEl.scrollTop = transcriptEl.scrollHeight;
    if (!transcriptExpanded) {
      transcriptExpanded = true;
      transcriptEl.classList.add("visible");
      expandBtn.textContent = "▲";
    }
  }

  // --- Drag ---

  let isDragging = false;
  let dragOffsetX = 0;
  let dragOffsetY = 0;

  pill.addEventListener("mousedown", (e) => {
    if (e.target === btn || e.target === expandBtn) return;
    isDragging = true;
    pill.classList.add("dragging");
    dragOffsetX = e.clientX - overlay.getBoundingClientRect().left;
    dragOffsetY = e.clientY - overlay.getBoundingClientRect().top;
    e.preventDefault();
  });

  document.addEventListener("mousemove", (e) => {
    if (!isDragging) return;
    overlay.style.left = `${e.clientX - dragOffsetX}px`;
    overlay.style.top = `${e.clientY - dragOffsetY}px`;
    overlay.style.right = "auto";
  });

  document.addEventListener("mouseup", () => {
    isDragging = false;
    pill.classList.remove("dragging");
  });

  syncState();
})();
