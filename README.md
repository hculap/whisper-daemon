# whisper-daemon

Local dictation and meeting recording for macOS on Apple Silicon. Whisper runs
on the Mac; microphone and browser audio are saved continuously before voice
filtering or transcription. The Chrome extension supports Google Meet participant
audio. Other applications can use an audio loopback device such as BlackHole.

## Install and prepare

Requires Apple Silicon, Python 3.11+, macOS 13+, and `ffmpeg` on PATH. Model files
and the optional speaker pipeline take substantially more space than the Python
source; pyannote also installs PyTorch.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
brew install ffmpeg
whisper-daemon models prepare
whisper-daemon models check
whisper-daemon start
```

Grant Microphone and Accessibility permissions to the app running the daemon.
Screen Recording permission is needed only when screenshots are enabled.

Inference is offline by default: missing Whisper/VAD models require explicit preparation. Hugging Face downloads and telemetry are disabled,
as is pyannote usage telemetry. `models prepare` is the explicit online provisioning
step. Missing models cause an actionable error instead of an implicit download.
For speaker recognition, obtain access to
[pyannote community-1](https://huggingface.co/pyannote/speaker-diarization-community-1),
sign in using `hf auth login`, then run:

```sh
whisper-daemon models prepare --diarization
whisper-daemon models check
```

The app checks that the speaker pipeline and dependencies actually load before
capture. If unavailable, it shows a warning and saves audio/transcripts without
automatic speaker labels. No token is included in recordings or the repository.

## Dictation

| Action | Default shortcut |
| --- | --- |
| Start / stop | Cmd+Shift+Space |
| Cancel recording, inference or draft | Cmd+Shift+Escape |
| Accept a reviewed draft | Cmd+Shift+Enter |
| Paste last result | Cmd+Shift+V |
| Speak selected text | Cmd+Shift+C |

The menu shows a live preview. **Dictation Settings** controls PL/EN/automatic
language, silence duration (0 means manual stop), vocabulary hints, confirmation
before pasting, and all five shortcuts. **Review / Edit Dictation** edits a completed
draft when confirmation is enabled. Saving an edit keeps the draft: return to
the original input field and press the accept shortcut to paste. Shortcuts are reserved globally; choose
combinations that do not conflict with your applications.

The final transcription includes the entire captured audio, including speech after
the last preview. Automatic paste checks the original app, window and focused
input. If that target changed or cannot be verified, the result stays in **Recent**.
Recent entries can be pasted again and survive restart (up to 100 entries in
`~/.config/whisper-daemon/dictation-history.json`, file mode 0600). Paste preserves
all clipboard formats and does not overwrite a new copy made during the paste.

## Chrome pairing and recording

1. Load `extension/` as an unpacked extension in `chrome://extensions`.
2. Open its popup. Run the displayed `whisper-daemon pair EXTENSION_ID` command.
3. Paste the returned private key into the popup and save it.
4. Start recording from the Meet button or extension popup.

Pairing permits only that exact extension ID and a secret held in trusted extension
storage. The daemon binds only to loopback. Reload the extension and restart the
daemon when upgrading: the authenticated, sequenced protocol is incompatible with
older versions. Pairing another extension rotates the key.

Microphone and participant indicators are separate. The native menu and extension
popup/panel provide **Pause / Resume**; paused intervals retain their position in
the timeline as silence. Screenshots pause too. Stop finishes the pending disk
queue. Another meeting cannot start until finalization has ended.

Packet sequence and source sample positions detect duplicates and missing audio.
Reconnects resume the same meeting. A transport loss leaves microphone capture
running and the participant indicator reports loss; use the native Stop control
if the browser is unreachable. The extension reports buffer exhaustion explicitly.
Stop during reconnect first drains retained audio. If the session was already
closed from the native menu, the extension popup offers a local WAV export of its
undelivered tail, including timing gaps and the starting offset in the filename.
Download it before closing Chrome; this exceptional tail is retained in extension
memory, not in the daemon's already finalized recording. Starting another session
does not discard it. Remove the buffer copy only after checking the downloaded file.

## Files, recovery and meeting review

Each meeting folder contains:

- `mic.f32` / `browser.f32`: continuous 16 kHz mono float32-LE sources, including silence.
- `mic.wav` / `browser.wav`: aligned source exports; `recording.wav`: combined playback.
- `session.json`: duration, missing-source intervals, pauses and errors.
- `chunks/`: durable pending jobs and completed results.
- `transcript.json` plus selected text/subtitle formats.
- `notes.json`: editable summary and task suggestions with source timestamps.
- `meeting.ready.json` or `meeting.failed.json`: the Orbit readiness marker (see below).

Audio is always retained for recovery. The old optional `save_audio` setting no
longer disables source capture. Allow about 810 MB/hour for two raw sources plus
source and mixed WAVs (pending jobs and screenshots need additional space).
Files are written before inference; disk errors are surfaced and stop capture.
Raw writes are unbuffered and fsynced once per second. Process-crash recovery uses
file lengths rather than a WAV header; a power failure may lose the last unsynced
second. There is no automatic deletion or alteration of older recordings.

```sh
whisper-daemon recover /path/to/interrupted/recording
whisper-daemon recover /path/to/interrupted/recording --transcribe --language pl
whisper-daemon review /path/to/recording
```

The native menu also has **Review Last Meeting** and **Open Meeting**. The local
review page supports search, timestamp playback, source selection, speaker names,
per-utterance speaker corrections, and adding speakers. Saving persists edits and
regenerates TXT/SRT/VTT while preserving `transcript_original.json`.

Initial notes are extractive suggestions, not an AI-generated account of facts:
selected utterances and action-like phrases retain their recording timestamps.
Verify tasks, fill in owners/dates, and mark completion. The review server is bound
to `127.0.0.1` and uses an unpredictable per-run URL; it loads no external assets.
Keep that URL private. Closing the daemon/CLI closes the review server.

## CLI

```sh
whisper-daemon run                         # foreground
whisper-daemon stop
whisper-daemon status
whisper-daemon logs -f
whisper-daemon devices
whisper-daemon record meeting.txt --format txt,srt,json --language pl
whisper-daemon record meeting.txt --device 'BlackHole 2ch'
whisper-daemon transcribe meeting.mp3 --format txt,srt,json
```

CLI recording ends with Ctrl+C and saves continuous audio in an adjacent unique
`*_audio_*` folder. Native/browser speaker recognition runs after capture; the CLI
also retains experimental realtime/hybrid speaker tracking options.

## Orbit: automatic meeting processing

Every meeting folder carries a readiness contract for Orbit, the server-side
meeting analyzer. Recording starts by writing `meeting.in-progress.json`.
`meeting.ready.json` is written atomically and **last**: after every chunk job is
acknowledged and the WAV exports, screenshots, `transcript.json`, selected formats,
`notes.json` and the final `session.json` are on disk. It holds a stable meeting ID,
a revision, the start time with timezone, the transcription status and the
SHA-256, size and role (`audio`, `transcript` or `attachment`) of every published
file. Raw `*.f32` sources and `chunks/` are local recovery material and are not
listed. An empty transcript, a failed chunk or a failed recording writes
`meeting.failed.json` instead; a killed process leaves only
`meeting.in-progress.json`. Neither ever gets the ready marker. A published
revision is never rewritten; each recording uses a new folder.

Orbit reads `<synced root>/*/meeting.ready.json`, so every meeting must be a direct
child of the synced folder: point the menu-bar recording directory at it. CLI
recordings opt in:

```sh
whisper-daemon record ~/Sync/Recordings/meeting.txt --orbit-ready --format txt,srt
```

Instead of `meeting.txt` plus a `*_audio_*` folder, this saves the continuous audio,
`transcript.json` (always, whatever `--format` says), the selected `transcript.*`
formats and the manifest in one new `recording_*` folder next to the requested
output. Orbit verifies every listed file after sync (the marker may arrive first),
archives the meeting and analyzes it; see Orbit's `docs/MEETING_AUTOMATION.md`.
The contract tests run without a microphone, GPU or Cocoa
(`tests/test_meeting_manifest.py`, `tests/test_record_manifest.py`,
`tests/test_meeting_lifecycle.py`); capture and sync must still be verified on macOS.

## Verification

```sh
python -m pytest -q
node --test extension/test/*.test.js
python scripts/verify_110_minutes.py
python scripts/verify_speech_e2e.py  # macOS voices + cached Whisper/Silero, offline
```

The Node suite includes `daemon-e2e.test.js`: the real background and offscreen
scripts over real WebSockets against `tests/daemon_harness.py`, which runs the real
bridge and meeting pipeline headless (devices, VAD and Whisper stubbed). It needs
the project `.venv` (or `WD_E2E_PYTHON`).

The 110-minute test feeds both real recorder/storage pipelines at accelerated speed
and checks exact sample counts, silent intervals, WAV lengths and bounded memory.
It is separate from a 110-minute wall-clock microphone/Meet soak test.
See [delivery evidence](docs/delivery-plan.md) for executed checks and limitations.
