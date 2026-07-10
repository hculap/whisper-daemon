/**
 * Content script for meet.google.com — detects meeting join/leave.
 *
 * Uses MutationObserver to watch for the "Leave call" button (ARIA label)
 * which appears when the user joins a meeting.
 */

let meetingActive = false;
let debounceTimer = null;
let pollTimer = null;

// After the extension is reloaded/updated, content scripts already running on
// open Meet tabs are ORPHANED: chrome.runtime is gone, so any sendMessage throws
// "Extension context invalidated". Detect that, stop all timers/observers, and
// bail quietly instead of spamming the console every 3s.
function contextValid() {
  return !!(chrome.runtime && chrome.runtime.id);
}

function stopAll() {
  try { observer.disconnect(); } catch {}
  if (pollTimer) clearInterval(pollTimer);
  if (debounceTimer) clearTimeout(debounceTimer);
  pollTimer = null;
}

function send(message) {
  if (!contextValid()) {
    stopAll();
    return;
  }
  try {
    chrome.runtime.sendMessage(message);
  } catch (err) {
    // Orphaned after a reload — stop cleanly.
    stopAll();
  }
}

function checkMeetingState() {
  if (!contextValid()) {
    stopAll();
    return;
  }

  const leaveBtn = document.querySelector(
    '[aria-label*="Leave call"], [aria-label*="Leave the call"], [aria-label*="Opuść"]'
  );
  const isActive = leaveBtn !== null;

  if (isActive && !meetingActive) {
    meetingActive = true;

    const title =
      document.querySelector("[data-meeting-title]")?.getAttribute("data-meeting-title") ||
      document.title.replace(" - Google Meet", "").trim();

    send({ type: "MEET_JOINED", title, url: location.href });
  } else if (!isActive && meetingActive) {
    meetingActive = false;
    send({ type: "MEET_LEFT" });
  }
}

function debouncedCheck() {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(checkMeetingState, 500);
}

const observer = new MutationObserver(debouncedCheck);
if (document.body) {
  observer.observe(document.body, { childList: true, subtree: true });
}

// Initial check + fallback polling (Meet is a SPA)
checkMeetingState();
pollTimer = setInterval(checkMeetingState, 3000);
