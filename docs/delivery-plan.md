# Reliable dictation and meetings

Branch: `feat/reliable-dictation-meetings`; review base: `55b75ca`.

## Delivery gates

Each implementation milestone has a regression test (observed failing before
the fix), focused validation, and a separate commit. Production recordings and
the running daemon are preserved until an explicit integration check.

| Milestone | Scope | State |
| --- | --- | --- |
| M1 | B01 double PCM; B02 merge loss; B06 empty CLI; B07 dictation tail; B08 clipboard; B09 extension races | implemented; 103 Python + 25 JS tests pass |
| M2 | B03 continuous source audio; B04 crash recovery; B05 session ownership; bounded queues, pause/resume, 110-minute capture | implemented; 111 Python tests pass |
| M3 | Paired/authenticated bridge, validated sequenced audio; offline models and diarization preflight | implemented; 122 Python + 25 JS tests pass |
| M4 | Dictation preview/cancel, configurable shortcuts and silence, languages, vocabulary, persistent history, focus protection | implemented; 133 Python tests pass |
| M5 | Source health indicators, searchable playback, speaker editing, local summaries and action items | implemented; 136 Python + 27 JS tests pass; UI verification follows |
| M6 | PR, independent subagent review, corrections, final end-to-end verification | implemented; 153 Python + 38 JS tests pass; evidence below |

## Acceptance evidence

- No missing tail, no cross-source deletion, no duplicate browser PCM.
- Continuous per-source audio survives a killed writer; source gaps are explicit.
- 110 minutes of synthetic samples retain exact duration without RAM growth
  proportional to meeting length. Accelerated testing is distinguished from
  wall-clock capture and real microphone/browser checks.
- Controls are session-scoped; finalization, file jobs and reconnects cannot
  stop or overwrite another session. A requested stop cannot start recording.
- Unpaired clients cannot read results, alter settings, or start recording.
- Offline mode prevents implicit downloads and disables dependency telemetry;
  missing diarization models are visible before capture, with plain recording
  remaining available.
- Dictation delivery preserves the clipboard and checks the intended target.
- Meeting review works locally, with timestamped playback and editable speakers.
- Existing recordings are never migrated or overwritten implicitly.

## Audit findings

All B01–B09 and the three additional findings are implemented in M1–M5 and refined
by independent review. External model-account permissions cannot be fabricated:
actual speaker-model inference remains unverified because its authorized model is
not available locally. Readiness detection and fallback have been verified.

### M1 evidence

Regression failures reproduced before fixes: cross-source transcript loss, lost overlap tail, empty CLI text, lost dictation tail, rich clipboard destruction, duplicate PCM, pending-start and daemon-stop races. Full suite: 103 Python tests and 25 Node tests pass. Source timeline propagation and reconnect identity are completed in M2/M3.

### M2 evidence

`verify_110_minutes.py`: accelerated 6,600 seconds through two real recorder/store pipelines; 105,600,000 samples per source, exact WAV lengths and spot-checked content. Runtime 29.46 s, peak RSS growth 17,956,864 bytes. This is not a 110-minute wall-clock microphone/browser test. Abrupt `os._exit(9)` recovery, silent audio and final 513-sample tail tests pass. Disk-backed pending jobs survive reconstruction; only one inference job is held in RAM. All raw sources are retained independently of VAD and transcription success.

### M3 evidence

Real loopback WebSocket test rejects an unpaired origin, wrong secret and malformed JSON; reconnect replays are idempotent and missing PCM is filled/marked. Missing-model check runs with socket.connect forbidden. `models check` confirms local Whisper cache, unavailable diarization model, and enforced offline/telemetry policy. Gated model access is still an external requirement; no credentials or recordings were uploaded. Extension now requires explicit pairing; protocol is incompatible with old extension versions.

### M4 evidence

Cancellation generation test rejects a late result after another dictation starts. Focus rejection preserves text in persistent history without touching the clipboard. Review mode needs an explicit accept and supports edited text. Settings roundtrip covers PL, vocabulary, pause and custom shortcut; native AX probe can identify the current input. History is local, limited to 100 entries and mode 0600. Native menu exposes all controls.

### M5 evidence

Local HTTP integration verifies byte-range playback, private routes, cross-origin write rejection, speaker reassignment, new speaker creation and persisted exports. Extractive task suggestions keep source timestamps and require confirmation; no model-invented owners/dates. RTC graph start/stop tests cover positioned PCM and stop during module loading. Screenshot capture observes meeting pause.

### M6 independent review and corrections

Three independent reviewers examined audio persistence/lifecycle, extension/bridge,
and dictation/review UI. Their reproducible findings were corrected with regression
tests and re-reviewed. No findings were waived as accepted risk.

| Findings | Severity | Resolution / evidence |
| --- | --- | --- |
| A01 replay arrival-time drift; A02 paused replay retained | BLOCKER | Capture positions reach raw storage; capture-time pause mask excludes buffered private samples; exact replay and odd packet boundary tests |
| A03 silently dead microphone | BLOCKER | Callback liveness requests recovery, warns and marks gaps; worker reproduction verified |
| A04 one-sample tail; A05 trailing missing source | DEBT | Tail remains 1D; enabled sources receive explicit trailing gap intervals |
| A06 pause during preflight; A07 stop during initialization | BLOCKER | Timestamped pending pauses are handed off under lock; stopped worker drains acknowledged PCM without opening new devices; final duration uses stop time |
| B-EXT-1 reconnect ordering; B-EXT-2 stop during start; B-EXT-3 stale idle | BLOCKER | ACK/replay readiness, session-scoped termination, explicit stop during pending start; VM and real WebSocket reproductions verified |
| D-EXT-1 settings in second tab; D-EXT-2 stale ownership | DEBT | Read-only settings broadcast; terminal idle releases ownership; native controls work without browser ownership |
| Stop during reconnect and after native session close | BLOCKER | Retained backlog drains before stop over shared connection; ended-session tail can be exported locally without blocking new capture |
| R-D1 second save loses UI edits; R-D3 implicit VAD download | BLOCKER | Revision-aware save/rebind; VAD prepared explicitly and missing cache fails offline |
| R-D2 CLI silence overridden; R-D4 cancelled preview inference | DEBT | Explicit runtime override; generation check after model lock acquisition |
| Native editor focus lead | Preventive correction | Editing keeps REVIEW state; return to original field to accept; generation-scoped edit regression |
| CLI speaker model readiness / later failure | Preventive correction | Same visible preflight as native capture; failure preserves plain transcript; simulated 401 regression |
| CLI realtime embedding exception swallowed | DEBT | Propagated error selects final batch labels; full CLI regression verifies one batch call, visible fallback and retained TXT/JSON |

Final accelerated capture now traverses the actual loopback WebSocket and native
ingress method, not just direct store calls: 6,600 seconds and 105,600,000 samples
per source, ten-second delayed replay plus duplicate packet, no missing intervals,
14,254,080-byte RSS growth, 60.88-second runtime. Synthetic PCM and VAD replacement
are explicit; this is not a 110-minute physical meeting.

`verify_speech_e2e.py` generates PL/EN speech using macOS voices and runs actual
Silero VAD, the authenticated bridge, disk spool, cached MLX Whisper, source merge
and review exports with external socket connections forbidden. Both final “42”
sentences survive. Four segments / 17.2981875-second timeline; runtime 8.47 seconds.

Chrome UI verification: search and timestamp playback, speaker/task editing, two
successive saves and reload preserve the second edits. Screenshot evidence retained
locally with synthetic content. Physical default-microphone smoke test received
and saved about five seconds, with initial/trailing device latency marked as gaps;
its audio was deleted with the isolated temporary test directory.

Packaging: offline wheel build includes all three local review assets. The
production daemon and installed Chrome extension were not restarted or upgraded.
No actual 110-minute call, installed-extension tabCapture, permission-dialog flow,
or authorized diarization inference was claimed by these checks.

### M7 review corrections and end-to-end harness

`tests/daemon_harness.py` runs the real bridge (thread, loop, pairing, origin
check) and the real menubar connect/stage/stop/status/meeting-worker/store code;
only AppKit UI, devices, VAD and Whisper are stubbed, with configurable main-thread
latency, transcription and finalization time. `extension/test/daemon-e2e.test.js`
loads the real `background.js` + `offscreen.js` on a Chrome-like message bus and
drives them over real WebSockets against it, including daemon kill/restart and
refused connections. Every row below failed before its fix.

| Finding | Resolution / evidence |
| --- | --- |
| Settings snapshot or saved recording released the joined Meet tab; leaving the call then kept the socket open | Joined tab survives idle statuses; snapshots never override a start; native recordings use a flag. e2e: socket closes and `WD_MEET_LEFT` arrives after both |
| Start after a retained tail was refused for 10 s; a busy refusal closed the session and socket | Daemon answers `busy` without closing the session or socket; extension keeps buffering and re-sends `start` on idle/retry, keeping audio if it must give up. e2e: second meeting saved with all buffered audio |
| Restarted daemon dropped a recovered tail (stop handled before the main thread started it) | Stopped-before-start session with staged audio is saved browser-only, no devices; later starts are `busy` while one is pending. e2e with 200 ms main-thread latency + pytest |
| Unexported tail counted against the live buffer | Live budget covers live audio only. e2e: 1.9 s tail, 2 s budget, next recording unaffected |
| Background stuck in "saving" | No local refusal; closed control socket and any idle release it. e2e: daemon killed while saving |
| Late status cancelled `stopCapture` teardown; superseded popup start left STARTING | Generation bumps only for live sessions; every start settles background. Deterministic VM reproductions |
| Stop during reconnect gave up after one shared attempt | Joining another caller's attempt costs one try. e2e: leaving during a refused reconnect delivers the tail |
| Final `stop` lost when the client closed immediately (reply raised on the closed peer) | Replies tolerate a closed peer; queued frames are still handled. e2e failed 4/5 before, 0/5 after; pytest |
| Phantom trailing gaps on healthy recordings | Trailing shortfall up to 0.25 s is latency, not missing audio. e2e: 1.5 s real-time recording, `gaps == []` |

Script conventions (shebang, logging, type hints) were corrected. Final run: 158
Python and 53 Node tests (8 e2e scenarios plus the mid-recording leave) pass three
consecutive times; `verify_110_minutes.py` 6,600 s per source, no gaps, 19,431,424-byte
RSS growth, 60.9 s; `verify_speech_e2e.py` preserves both "42" sentences in 10.6 s.

Live Chrome check after merge: a paired extension never connected. The offscreen
document read the pairing key with `chrome.storage`, but offscreen documents only
get `chrome.runtime`; the read threw and was reported as "daemon unreachable". The
test fakes had given the offscreen realm a storage API, so 21 tests passed falsely.
The service worker now hands the key only to its own offscreen document, and the
fakes match Chrome (21 failed before the fix, 54/54 pass after). In a real Meet call
the settings panel loaded, a 43 s recording and a leave-mid-recording session both
saved `complete`, and leaving closed the socket.
