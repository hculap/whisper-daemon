/**
 * Content script for meet.google.com — detects meeting join/leave.
 *
 * Uses MutationObserver to watch for the "Leave call" button (ARIA label)
 * which appears when the user joins a meeting.
 */

let meetingActive = false;
let debounceTimer = null;

function checkMeetingState() {
  const leaveBtn = document.querySelector(
    '[aria-label*="Leave call"], [aria-label*="Leave the call"], [aria-label*="Opuść"]'
  );
  const isActive = leaveBtn !== null;

  if (isActive && !meetingActive) {
    meetingActive = true;

    const title =
      document.querySelector("[data-meeting-title]")?.getAttribute("data-meeting-title") ||
      document.title.replace(" - Google Meet", "").trim();

    chrome.runtime.sendMessage({
      type: "MEET_JOINED",
      title,
      url: location.href,
    });

  } else if (!isActive && meetingActive) {
    meetingActive = false;
    chrome.runtime.sendMessage({ type: "MEET_LEFT" });
  }
}

function debouncedCheck() {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(checkMeetingState, 500);
}

const observer = new MutationObserver(debouncedCheck);
observer.observe(document.body, { childList: true, subtree: true });

// Initial check + fallback polling (Meet is a SPA)
checkMeetingState();
setInterval(checkMeetingState, 3000);
