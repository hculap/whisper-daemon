const statusDot = document.getElementById("statusDot");
const meetingInfo = document.getElementById("meetingInfo");
const timer = document.getElementById("timer");
const actionBtn = document.getElementById("actionBtn");
const transcript = document.getElementById("transcript");

let timerInterval = null;
let recordingStart = null;

// --- Init ---

chrome.runtime.sendMessage({ type: "GET_STATE" }, (response) => {
  if (!response) return;
  updateUI(response.state, response.meetTitle);
});

// --- Live updates ---

chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type === "CAPTURE_STARTED") {
    updateUI("capturing");
  } else if (msg.type === "CAPTURE_STOPPED" || msg.type === "CAPTURE_ERROR") {
    updateUI("idle");
  } else if (msg.type === "CHUNK_TRANSCRIBED") {
    addTranscriptChunk(msg.text);
  }
});

// --- Button ---

actionBtn.addEventListener("click", async () => {
  const response = await new Promise((resolve) => {
    chrome.runtime.sendMessage({ type: "GET_STATE" }, resolve);
  });
  if (!response) return;

  if (response.state === "capturing") {
    chrome.runtime.sendMessage({ type: "STOP_RECORDING" });
    return;
  }

  if (response.state === "starting") return;

  // Get tab to capture
  const tabId = response.meetTabId || (await getActiveTabId());
  if (!tabId) return;

  updateUI("starting");

  // Call tabCapture from popup (has user gesture / activeTab grant)
  try {
    const streamId = await new Promise((resolve, reject) => {
      chrome.tabCapture.getMediaStreamId({ targetTabId: tabId }, (id) => {
        if (chrome.runtime.lastError) {
          return reject(new Error(chrome.runtime.lastError.message));
        }
        if (!id) return reject(new Error("Empty stream ID"));
        resolve(id);
      });
    });

    chrome.runtime.sendMessage({
      type: "START_WITH_STREAM",
      streamId,
      tabId,
    });
  } catch (err) {
    console.error("tabCapture failed:", err);
    updateUI("idle");
  }
});

async function getActiveTabId() {
  const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
  return tabs[0]?.id ?? null;
}

// --- UI ---

function updateUI(state, title) {
  statusDot.className = `status-dot ${state}`;

  if (state === "capturing") {
    actionBtn.textContent = "Stop Capture";
    actionBtn.className = "btn btn-stop";
    actionBtn.disabled = false;
    startTimer();
    transcript.classList.add("visible");
  } else if (state === "starting") {
    actionBtn.textContent = "Starting...";
    actionBtn.className = "btn btn-record";
    actionBtn.disabled = true;
  } else {
    actionBtn.textContent = state === "detected" ? "Start Capture" : "Capture This Tab";
    actionBtn.className = "btn btn-record";
    actionBtn.disabled = false;
    stopTimer();
  }

  if (title && state !== "idle") {
    meetingInfo.textContent = title;
    meetingInfo.classList.add("visible");
  } else {
    meetingInfo.classList.remove("visible");
  }
}

function addTranscriptChunk(text) {
  const div = document.createElement("div");
  div.className = "transcript-chunk";
  div.textContent = text;
  transcript.appendChild(div);
  transcript.scrollTop = transcript.scrollHeight;
}

// --- Timer ---

function startTimer() {
  recordingStart = Date.now();
  timer.classList.add("visible");
  updateTimerDisplay();
  timerInterval = setInterval(updateTimerDisplay, 1000);
}

function stopTimer() {
  clearInterval(timerInterval);
  timerInterval = null;
  timer.classList.remove("visible");
}

function updateTimerDisplay() {
  if (!recordingStart) return;
  const elapsed = Math.floor((Date.now() - recordingStart) / 1000);
  const minutes = String(Math.floor(elapsed / 60)).padStart(2, "0");
  const seconds = String(elapsed % 60).padStart(2, "0");
  timer.textContent = `${minutes}:${seconds}`;
}
