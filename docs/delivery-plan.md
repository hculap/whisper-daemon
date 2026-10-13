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
| M6 | PR, independent subagent review, corrections, final end-to-end verification | pending |

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

All B01–B09 and the three additional findings are open until their changes and
regression checks land. External model-account permissions cannot be fabricated;
their actual verification state must remain explicit.

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
