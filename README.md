# whisper-daemon

Local push-to-talk transcription daemon for macOS (Apple Silicon) — record with a hotkey, transcribe offline via `mlx-whisper`, paste text into the active field.

## Features

- **Fully offline** — everything runs locally, no data leaves your machine
- **Fast** — 4-bit quantized `whisper-large-v3-turbo` via MLX, ~1s transcription latency
- **Push-to-talk** — global hotkey `Cmd+Shift+Space` with automatic silence detection (Silero VAD)
- **Paste-to-focus** — transcription lands in the active text field, any app
- **Meeting recording** — record from mic or system audio (BlackHole), chunked parallel transcription
- **File transcription** — batch transcribe audio/video files to txt, srt, vtt, json
- **Menu bar** — status icon showing idle/recording/transcribing
- **Lightweight** — ~65MB install (no torch dependency)

## Install

```bash
git clone <repo-url> && cd whisper-daemon
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

Grant permissions: **System Settings > Privacy & Security > Accessibility + Microphone** (for Terminal or your IDE).

## Usage

### Push-to-talk daemon

```bash
whisper-daemon start              # background with menu bar
whisper-daemon start -v           # verbose logging
whisper-daemon stop               # stop daemon
whisper-daemon status             # check if running
whisper-daemon run                # foreground (debug)
whisper-daemon logs -f            # follow logs
```

Press `Cmd+Shift+Space` to record → speak → wait for silence → text appears.

### Transcribe files

```bash
whisper-daemon transcribe meeting.mp3
whisper-daemon transcribe meeting.mp3 --format srt
whisper-daemon transcribe meeting.mp3 --format txt,srt,json
whisper-daemon transcribe ./recordings/
whisper-daemon transcribe video.mp4 --output-dir ./out
```

### Record meetings

```bash
whisper-daemon record meeting.txt                                # from default mic
whisper-daemon record meeting.txt --device "BlackHole 2ch"       # system audio
whisper-daemon record meeting.txt --format txt,srt --language en
whisper-daemon devices                                           # list audio devices
```

Records until `Ctrl+C`. Splits audio at natural pauses and transcribes in parallel.

### System audio capture (Google Meet, Zoom)

To capture audio from other apps, install [BlackHole](https://github.com/ExistentialAudio/BlackHole):

```bash
brew install blackhole-2ch
```

Then in **Audio MIDI Setup** (built-in macOS app):

1. Create a **Multi-Output Device**: combine "Built-in Output" + "BlackHole 2ch"
2. Set the Multi-Output Device as your macOS sound output
3. Record: `whisper-daemon record meeting.txt --device "BlackHole 2ch"`

You'll hear audio through speakers AND it gets captured by the daemon.

## Requirements

- macOS 13+ (Ventura)
- Apple Silicon (M1/M2/M3/M4)
- Python 3.11+
- ffmpeg (for file transcription): `brew install ffmpeg`

## Documentation

- [Product Requirements](docs/prd.md)
- [Architecture](docs/architecture.md)

## License

MIT
