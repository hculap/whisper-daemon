"""Post-processing to strip Whisper hallucination loops from transcripts.

On low-information audio (near-silence, background noise, a trailing breath),
Whisper's decoder gets stuck repeating a short token or phrase — "do do do do",
"Nie, nie, nie, ..." (x100), "það er það er", "Dwa... Dwa...". Temperature
fallback / compression-ratio thresholds catch some of it probabilistically but
miss trailing runs inside otherwise-valid segments, so this deterministic pass
collapses any short phrase (1-8 tokens, e.g. "że to jest to," x43) repeated 3+
times to a single instance, comparing case- and punctuation-insensitively while
preserving the original tokens. Genuine speech (no long verbatim repeats) is
left untouched. ``clean_segments`` also drops whole hallucinated segments.
"""

from itertools import groupby
import re

_MAX_PHRASE_TOKENS = 8
_MIN_REPEATS = 3


def _norm(token: str) -> str:
    """Lowercase and strip non-word chars for repeat comparison."""
    return re.sub(r"[^\w]", "", token.lower())


def collapse_repetitions(text: str) -> str:
    """Collapse short phrases repeated ``_MIN_REPEATS``+ times to one copy.

    Comparison ignores case/punctuation so "Nie, nie, nie" and "do Do Do" are
    caught; the kept copy keeps its original spelling. All-punctuation phrases
    (e.g. "... ... ...") are skipped by the phrase-length loop and handled as a
    final trim.
    """
    if not text:
        return text

    tokens = text.split()
    n = len(tokens)
    out: list[str] = []
    i = 0
    while i < n:
        matched = False
        for plen in range(1, _MAX_PHRASE_TOKENS + 1):
            if i + plen > n:
                continue
            phrase_norm = [_norm(t) for t in tokens[i : i + plen]]
            if not any(phrase_norm):
                continue  # all-punctuation phrase — let the fall-through keep one
            reps = 1
            j = i + plen
            while j + plen <= n and [_norm(t) for t in tokens[j : j + plen]] == phrase_norm:
                reps += 1
                j += plen
            if reps >= _MIN_REPEATS:
                out.extend(tokens[i : i + plen])  # keep a single copy
                i = j
                matched = True
                break
        if not matched:
            out.append(tokens[i])
            i += 1

    return " ".join(out)


# Whole segments Whisper emits on noise or near-silence: subtitle credits and
# video outros from its training data (seen in real meetings), not speech.
_OUTROS = frozenset({
    "wszystkie prawa zastrzeżone", "napisy",
    "dziękuję za oglądanie", "dziękuje za oglądanie",
    "dziękuję za uwagę", "dziękuje za uwagę",
    "thanks for watching", "thank you for watching",
})


def _phrase(text: str) -> str:
    return " ".join(filter(None, map(_norm, text.split())))


def clean_segments(result: dict) -> dict:
    """Collapse loops in one transcription and drop its hallucinated segments.

    Dropped: segments without words ("", "..."), outros (``_OUTROS``, Amara.org
    credits), and all but the first of ``_MIN_REPEATS``+ identical segments in
    a row ("1", "1", "1"). The text is rebuilt from the kept segments. A
    collapsed segment loses its word timestamps, which no longer match its text.
    """
    segments = result.get("segments", [])
    if not segments:
        return {**result, "text": collapse_repetitions(result.get("text", "").strip())}
    candidates: list[tuple[str, dict]] = []
    for seg in segments:
        original = seg.get("text", "").strip()
        text = collapse_repetitions(original)
        phrase = _phrase(text)
        if not phrase or phrase in _OUTROS or "amaraorg" in phrase:
            continue
        cleaned = {**seg, "text": text}
        if text != original:
            cleaned.pop("words", None)
        candidates.append((phrase, cleaned))
    kept: list[dict] = []
    for _, group in groupby(candidates, key=lambda item: item[0]):
        run = [seg for _, seg in group]
        kept.extend(run[:1] if len(run) >= _MIN_REPEATS else run)
    return {**result, "segments": kept, "text": " ".join(seg["text"] for seg in kept)}
