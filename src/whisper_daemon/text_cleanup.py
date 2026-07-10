"""Post-processing to strip Whisper hallucination loops from transcripts.

On low-information audio (near-silence, background noise, a trailing breath),
Whisper's decoder gets stuck repeating a short token or phrase — "do do do do",
"Nie, nie, nie, ..." (x100), "það er það er", "Dwa... Dwa...". Temperature
fallback / compression-ratio thresholds catch some of it probabilistically but
miss trailing runs inside otherwise-valid segments, so this deterministic pass
collapses any short phrase (1-3 tokens) repeated 3+ times to a single instance,
comparing case- and punctuation-insensitively while preserving the original
tokens. Genuine speech (no long verbatim repeats) is left untouched.
"""

import re

_MAX_PHRASE_TOKENS = 3
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
