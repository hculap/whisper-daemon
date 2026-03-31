# whisper-daemon

Lokalny daemon push-to-talk dla macOS (Apple Silicon) — nagrywaj głos hotkeyem, transkrybuj offline przez `mlx-whisper` i wklej tekst do aktywnego pola.

## Problem

Pisanie jest wolne. Dyktowanie przez Siri/Google wymaga chmury, ma opóźnienia i problemy z prywatnością. Brakuje prostego narzędzia, które:

- działa **w pełni offline** na M1/M2/M3,
- reaguje na **globalny hotkey** (nawet gdy inne okno jest aktywne),
- **automatycznie wykrywa ciszę** i kończy nagrywanie,
- **wkleja transkrypcję** prosto do aktywnego inputa — jak natywna funkcja systemu.

## Rozwiązanie

Lekki daemon Python, który nasłuchuje `Cmd+Shift+Space`, nagrywa audio z mikrofonu, przepuszcza przez `mlx-whisper` (zoptymalizowany pod Apple Silicon GPU) i symuluje wklejenie tekstu do aktywnego pola.

## Kluczowe cechy

- **Zero chmury** — wszystko działa lokalnie, żadne dane nie opuszczają maszyny.
- **Najszybszy STT na M1** — `mlx-whisper` z modelem `whisper-large-v3-turbo` (~13 sek na transkrypcję vs ~27 sek w whisper.cpp).
- **Push-to-talk + VAD** — hotkey startuje nagrywanie, ponowne naciśnięcie lub cisza (Silero VAD) je kończy.
- **Streaming (opcjonalnie)** — tekst pojawia się w trakcie mówienia, nie dopiero po zakończeniu.
- **Paste-to-focus** — wynik ląduje w aktywnym polu tekstowym, niezależnie od aplikacji.

## Stack

| Komponent | Technologia |
|---|---|
| Transkrypcja | `mlx-whisper` + `whisper-large-v3-turbo` |
| Nagrywanie audio | `sounddevice` (PortAudio) |
| Wykrywanie ciszy | `silero-vad` (ONNX) |
| Globalny hotkey | `pynput` |
| Wklejanie tekstu | `pyobjc` (NSPasteboard + CGEvent) |
| Streaming (opcja) | chunked inference z mlx-whisper |

## Wymagania systemowe

- macOS 13+ (Ventura lub nowszy)
- Apple Silicon (M1/M2/M3/M4)
- Python 3.11+
- Uprawnienia: Accessibility + Microphone

## Dokumentacja

- [Product Requirements Document](docs/prd.md)
- [Architektura](docs/architecture.md)

## Status

🟡 Faza projektowania — dokumentacja i architektura. Kod jeszcze nie powstał.

## Licencja

MIT
