# Product Requirements Document — whisper-daemon

**Wersja:** 0.2 (PoC)
**Data:** 2026-03-31
**Autor:** Szymon

---

## 1. Cel produktu

Lekki, lokalny daemon macOS zamieniający mowę w tekst i wklejający go do aktywnego pola — bez chmury, bez opóźnień sieciowych, bez kompromisów prywatności.

Użytkownik naciska hotkey, mówi, a tekst pojawia się tam, gdzie stoi kursor — jak natywna funkcja systemu.

## 2. Grupa docelowa

Programiści i power-userzy macOS z Apple Silicon, którzy chcą dyktować tekst do dowolnej aplikacji (edytor kodu, Slack, przeglądarka, terminal) bez polegania na usługach chmurowych.

## 3. User stories (PoC scope)

**US-1: Podstawowa transkrypcja**
Jako użytkownik, naciskam `Cmd+Shift+Space`, mówię zdanie, naciskam ponownie — tekst pojawia się w aktywnym polu tekstowym.

**US-2: Auto-stop na ciszę**
Jako użytkownik, naciskam hotkey, mówię i po prostu przestaję mówić — daemon sam wykrywa ciszę (VAD), kończy nagrywanie i wkleja tekst.

**US-3: Wielojęzyczność**
Jako użytkownik polski/angielski, daemon automatycznie rozpoznaje język mówienia.

### Odroczone (post-PoC)

- US-4: Streaming (tekst w trakcie mówienia)
- US-5: Feedback wizualny (menu bar icon)
- US-6: Konfiguracja hotkeya przez plik
- US-7: Hold-to-talk mode

## 4. Wymagania funkcjonalne (PoC)

### 4.1 Nagrywanie audio

| ID | Wymaganie |
|---|---|
| F-1 | Globalny hotkey `Cmd+Shift+Space` startuje/stopuje nagrywanie (toggle) |
| F-2 | Nagrywanie z domyślnego mikrofonu systemowego (16kHz, mono, float32) |
| F-3 | VAD (Silero VAD, ONNX) automatycznie kończy nagrywanie po 1.5s ciszy |
| F-4 | VAD czeka na pierwszą detekcję głosu zanim zacznie odliczać ciszę |
| F-5 | Bufor audio w pamięci (RAM only, brak zapisu na dysk) |
| F-6 | Max czas nagrywania: 120s (safety limit, zapobiega OOM) |

### 4.2 Transkrypcja

| ID | Wymaganie |
|---|---|
| T-1 | Transkrypcja przez `mlx-whisper` 0.4.x z modelem `mlx-community/whisper-large-v3-turbo` |
| T-2 | Automatyczna detekcja języka (Whisper native) |
| T-3 | Post-processing: trim whitespace only (Whisper obsługuje interpunkcję i kapitalizację) |
| T-4 | Pusty wynik (szum, cisza) — pominięcie paste, log warning |

### 4.3 Output / wklejanie

| ID | Wymaganie |
|---|---|
| O-1 | Wklejenie transkrypcji: NSPasteboard + symulacja Cmd+V przez CGEvent |
| O-2 | Przywrócenie poprzedniej zawartości schowka po 200ms delay |

### 4.4 Feedback (PoC — minimal)

| ID | Wymaganie |
|---|---|
| U-1 | Log do stdout: stan (IDLE/RECORDING/TRANSCRIBING), wynik transkrypcji |
| U-2 | Log error na stderr: brak mikrofonu, brak uprawnień, błąd transkrypcji |

## 5. Wymagania niefunkcjonalne

| ID | Wymaganie |
|---|---|
| NF-1 | Latencja end-to-end (stop → tekst wklejony) < 3s dla wypowiedzi do 30s |
| NF-2 | Zero ruchu sieciowego (poza pierwszym pobraniem modelu z HuggingFace) |
| NF-3 | Graceful shutdown na SIGTERM/SIGINT — przerwanie nagrywania, cleanup |
| NF-4 | Error recovery: błąd transkrypcji → powrót do IDLE, log error |

## 6. Wymagania systemowe

- macOS 13+ (Ventura)
- Apple Silicon (M1/M2/M3/M4)
- Python 3.11+
- Uprawnienia: Accessibility (hotkey + paste), Microphone

## 7. Czego NIE robimy (PoC out of scope)

- GUI / menu bar icon (konfiguracja = stałe w kodzie lub env vars)
- Plik konfiguracyjny TOML
- Streaming / chunked inference
- Hold-to-talk mode
- Tryb "type" (keystroke simulation)
- launchd autostart
- Wsparcie Intel / Linux / Windows
- Tłumaczenie (tylko transkrypcja)
- Wybór mikrofonu (zawsze domyślny)
- `initial_prompt` (dodamy po testach)

## 8. Metryki sukcesu (PoC)

| Metryka | Cel |
|---|---|
| Latencja end-to-end (30s audio) | < 3s |
| Działa z polskim i angielskim | Subiektywna ocena jakości |
| Hotkey działa z dowolną aktywną aplikacją | Manual test |
| VAD poprawnie wykrywa ciszę | Manual test |

## 9. Ryzyka

| Ryzyko | Mitygacja |
|---|---|
| `pynput` nie łapie hotkeya w niektórych kontekstach | Fallback: CGEvent tap przez pyobjc (post-PoC) |
| `torch` dependency dla Silero VAD (~2GB) | Akceptujemy w PoC; alternatywa: `py-silero-vad-lite` (ONNX-only, no torch) |
| macOS blokuje Accessibility bez ręcznego grantu | Jasny komunikat w logach + README |
| Pierwsza transkrypcja wolna (model download) | Log informujący o pobieraniu modelu |
