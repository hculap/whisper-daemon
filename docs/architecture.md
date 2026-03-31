# Architektura — whisper-daemon

**Wersja:** 0.2 (PoC)
**Data:** 2026-03-31

---

## 1. Przegląd

whisper-daemon to jednoprocesowy daemon Python. Składa się z pięciu modułów połączonych wewnętrzną kolejką zdarzeń. Jeden proces, kilka wątków — bez serwera, bez socketa, bez networku.

```
┌─────────────────────────────────────────────────────────┐
│                   whisper-daemon (PoC)                   │
│                                                         │
│  ┌───────────┐    ┌───────────┐    ┌────────────────┐   │
│  │  Hotkey   │───▶│  Audio    │───▶│  Transcriber   │   │
│  │  Listener │    │  Recorder │    │  (mlx-whisper) │   │
│  │ (pynput)  │    │(sounddev) │    │                │   │
│  └───────────┘    └─────┬─────┘    └───────┬────────┘   │
│                         │                  │            │
│                    ┌────▼────┐        ┌────▼─────┐      │
│                    │  VAD    │        │  Paster  │      │
│                    │(Silero) │        │ (pyobjc) │      │
│                    └─────────┘        └──────────┘      │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │              Event Bus (queue.Queue)              │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## 2. Moduły

### 2.1 Hotkey Listener (`hotkey.py`)

Nasłuchiwanie globalnego hotkeya `Cmd+Shift+Space` i emitowanie `RECORD_TOGGLE`.

`pynput.keyboard.GlobalHotKeys` — rejestruje kombinację na poziomie systemu (wymaga Accessibility). Działa w dedykowanym wątku (`.start()` / `.stop()`).

Uwaga: `<ctrl>` i `<alt>` są broken na macOS z pynput — `<cmd>+<shift>` działa poprawnie.

### 2.2 Audio Recorder (`recorder.py`)

Nagrywanie audio z domyślnego mikrofonu do bufora w pamięci.

`sounddevice.InputStream` — callback-driven, non-blocking. Format: 16kHz, mono, float32 (wymagany przez Whisper). Dane w `list[np.ndarray]` — każdy callback robi `indata.copy()`.

Parametry: `samplerate=16000, channels=1, dtype='float32'`.

Safety: max 120s nagrywania (16000 * 120 * 4 bytes = ~7.3 MB).

### 2.3 VAD (`recorder.py` — inline w callback)

Wykrywanie ciszy i automatyczne zakończenie nagrywania.

Silero VAD (ONNX): `load_silero_vad(onnx=True)`. Inference na CPU w <1ms per 512-sample chunk (32ms audio).

Logika:
1. Czeka na pierwszą detekcję głosu (`speech_prob > 0.5`) — nie odlicza ciszy dopóki użytkownik nie zacznie mówić.
2. Po wykryciu głosu, liczy czas ciszy.
3. Po 1.5s ciągłej ciszy → emituje `RECORD_STOP`.
4. `model.reset_states()` po każdym nagraniu.

VAD działa inline w sounddevice callback — każdy chunk przechodzi przez model.

**Uwaga**: Silero VAD oczekuje `torch.Tensor` i chunków dokładnie 512 samples (16kHz). Konwersja `np.ndarray → torch.Tensor` w callbacku.

### 2.4 Transcriber (`transcriber.py`)

Zamiana bufora audio na tekst.

```python
import mlx_whisper

result = mlx_whisper.transcribe(
    audio_np,  # np.ndarray, 16kHz mono float32
    path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
)
text = result["text"].strip()
```

Model auto-downloaduje się z HuggingFace przy pierwszym wywołaniu i cachuje lokalnie. Brak osobnego `load_model()` — model ładowany na pierwszym `transcribe()`.

Latencja: ~1-2s dla 30s audio na M1.

Obsługa błędów: wyjątek z `transcribe()` → log error, powrót do IDLE.

### 2.5 Paster (`paster.py`)

Wklejenie tekstu do aktywnego pola.

Wymaga: `pyobjc-framework-Cocoa` (NSPasteboard) + `pyobjc-framework-Quartz` (CGEvent).

Flow:
1. `NSPasteboard.generalPasteboard()` — zapisz bieżącą zawartość
2. `pb.setString_forType_(text, NSPasteboardTypeString)`
3. `CGEventCreateKeyboardEvent(None, 9, True/False)` + `CGEventSetFlags(kCGEventFlagMaskCommand)` — symuluj Cmd+V (keycode 9 = 'v')
4. `time.sleep(0.2)` — czekaj na paste
5. Przywróć oryginalną zawartość schowka

## 3. Threading model (PoC)

```
Main Thread        ── Event loop (queue.get) + State Machine + signal handling
Hotkey Thread      ── pynput GlobalHotKeys (daemon thread)
Audio Thread       ── sounddevice PortAudio callback (managed by sounddevice)
Transcriber Thread ── mlx-whisper inference (spawned per transcription)
```

Main thread = event loop. Brak menu bar w PoC, więc main thread jest wolny.

## 4. State Machine

```
         RECORD_TOGGLE           RECORD_TOGGLE / VAD silence
IDLE ──────────────────▶ RECORDING ──────────────────────────▶ TRANSCRIBING
 ▲                                                                │
 │                         ERROR                                  │
 ├────────────────────────────────────────────────────────────────┘
 │                    TRANSCRIPTION_DONE                           │
 └────────────────────────────────────────────────────────────────┘
```

- `IDLE` → `RECORDING`: na RECORD_TOGGLE
- `RECORDING` → `TRANSCRIBING`: na RECORD_TOGGLE (manual) lub RECORD_STOP (VAD)
- `TRANSCRIBING` → `IDLE`: na TRANSCRIPTION_DONE lub ERROR
- Hotkey w trakcie `TRANSCRIBING`: ignorowany

## 5. Przepływ danych

```
1. User naciska Cmd+Shift+Space
2. Hotkey Listener → RECORD_TOGGLE → State Machine: IDLE → RECORDING
3. Audio Recorder: stream.start() → callback: chunks do bufora + VAD
4. VAD wykrywa 1.5s ciszy (lub user naciska hotkey ponownie) → RECORD_STOP
5. State Machine: RECORDING → TRANSCRIBING
6. stream.stop(), bufor → np.concatenate() → numpy array
7. Transcriber thread: mlx_whisper.transcribe(audio) → text
8. TRANSCRIPTION_DONE(text) → Paster
9. Paster: save clipboard → set text → Cmd+V → sleep(0.2) → restore clipboard
10. State Machine: TRANSCRIBING → IDLE
```

## 6. Event Bus

`queue.Queue` (stdlib) — thread-safe, zero zależności.

| Zdarzenie | Producer | Consumer |
|---|---|---|
| `RECORD_TOGGLE` | Hotkey Listener | State Machine |
| `RECORD_STOP` | VAD / max time | State Machine |
| `TRANSCRIPTION_DONE(text)` | Transcriber | Paster → IDLE |
| `ERROR(msg)` | Any | State Machine → IDLE |

Zdarzenia jako `dataclass` lub `NamedTuple` z typem i opcjonalnym payload.

## 7. Struktura plików

```
whisper-daemon/
├── README.md
├── pyproject.toml
├── docs/
│   ├── prd.md
│   └── architecture.md
├── src/
│   └── whisper_daemon/
│       ├── __init__.py
│       ├── __main__.py      # entry point, arg parsing
│       ├── daemon.py         # state machine + event loop
│       ├── hotkey.py         # pynput global hotkey
│       ├── recorder.py       # sounddevice + Silero VAD
│       ├── transcriber.py    # mlx-whisper wrapper
│       └── paster.py         # NSPasteboard + CGEvent paste
└── tests/                    # post-PoC
```

## 8. Zależności

| Pakiet | Wersja | Cel |
|---|---|---|
| `mlx-whisper` | ≥0.4 | Transkrypcja (MLX backend) |
| `sounddevice` | ≥0.5 | Nagrywanie audio |
| `numpy` | ≥1.24 | Bufor audio |
| `pynput` | ≥1.7 | Globalny hotkey |
| `pyobjc-framework-Cocoa` | ≥10.0 | NSPasteboard |
| `pyobjc-framework-Quartz` | ≥10.0 | CGEvent (paste simulation) |
| `silero-vad` | ≥5.0 | Voice Activity Detection |
| `torch` | ≥2.0 | Runtime dla Silero VAD |
| `onnxruntime` | ≥1.16 | ONNX backend dla Silero VAD |

Uwaga: `torch` to ciężka zależność (~2GB). Post-PoC rozważyć `py-silero-vad-lite` (ONNX-only, bez torch).

## 9. Decyzje architektoniczne

**mlx-whisper > whisper.cpp** — 2x szybszy na Apple Silicon. Natywny Python, zero FFI.

**pynput > CGEvent tap** — prostsze API, wystarczy dla PoC. CGEvent tap jako fallback post-PoC.

**queue.Queue > asyncio** — mlx-whisper jest synchroniczny i CPU/GPU-bound. Threading z Queue prostszy.

**Silero VAD ONNX > webrtcvad** — dokładniejszy, lepszy dla wielojęzyczności. ONNX runtime nie koliduje z MLX.

**RAM buffer > temp file** — prywatność. 120s nagrania = ~7.3 MB RAM.

**Inline VAD > osobny wątek** — VAD inference <1ms, nie spowalnia audio callback.
