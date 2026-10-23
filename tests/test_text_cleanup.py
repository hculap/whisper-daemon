"""Tests for Whisper hallucination-loop collapsing."""

import pytest

from whisper_daemon.text_cleanup import clean_segments, collapse_repetitions


@pytest.mark.parametrize(
    "text,expected",
    [
        ("do do do do do do do", "do"),
        ("ja ja ja ja ja", "ja"),
        ("Nie, nie, nie, nie, nie, nie, nie", "Nie,"),
        ("Dwa... Dwa... Dwa... Dwa...", "Dwa..."),
        ("er það er það er það er það", "er það"),
        # trailing hallucination on otherwise-real speech: keep the speech
        ("no to po prostu będę do Do Do Do", "no to po prostu będę do"),
        # real meeting loops of 4-5 token phrases, cut at the 224-token limit
        ("Dziś, że to jest to, że to jest to, że to jest to, że to jest to, że to", "Dziś, że to jest to, że to"),
        ("Wydaje się, że to nie jest to, że to nie jest to, że to nie jest to,", "Wydaje się, że to nie jest to,"),
    ],
)
def test_collapses_hallucination_loops(text, expected):
    assert collapse_repetitions(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Sorry, taki deep focus wszedłem, że straciłem rachubę czasu.",
        "W sensie to proszę powiedzieć, że danie to się układa bardzo takie.",
        "No proszę, no więc taki status, ja się zbieram i tam jadę.",
        # a legitimate double is below the 3x threshold — must survive
        "bardzo bardzo dobrze to wyszło",
        "to jest to, to jest to, czego szukałem",
    ],
)
def test_preserves_real_speech(text):
    assert collapse_repetitions(text) == text


def test_empty_and_whitespace():
    assert collapse_repetitions("") == ""
    assert collapse_repetitions("   ") == ""


def seg(text, **extra):
    return {"start": 0.0, "end": 1.0, "text": text, **extra}


def test_clean_segments_drops_hallucinated_segments_and_rebuilds_text():
    result = {"language": "pl", "text": "raw", "segments": [
        seg(" ..."), seg(""), seg(" Coś wysyłam ci tutaj na czacie."),
        seg(" 1"), seg(" 1"), seg(""), seg(" 1"),
        seg(" Wszystkie prawa zastrzeżone."), seg(" Dziękuje za oglądanie!"),
        seg(" Napisy stworzone przez społeczność Amara.org"), seg(" Dobra, to wszystko."),
        seg(" 13 minut."), seg(" 13 minut."),  # a real double stays
    ]}
    cleaned = clean_segments(result)
    assert [s["text"] for s in cleaned["segments"]] == [
        "Coś wysyłam ci tutaj na czacie.", "1", "Dobra, to wszystko.", "13 minut.", "13 minut."]
    assert cleaned["text"] == "Coś wysyłam ci tutaj na czacie. 1 Dobra, to wszystko. 13 minut. 13 minut."
    assert cleaned["language"] == "pl"
    assert result["segments"][2]["text"] == " Coś wysyłam ci tutaj na czacie."  # input untouched


def test_a_collapsed_segment_drops_its_word_times():
    words = [{"word": " tak,", "start": 0.1 * i, "end": 0.1 * i + 0.1} for i in range(5)]
    loop, real = seg(" Tak, tak, tak, tak, tak,", words=words), seg(" Dobrze.", words=words[:1])
    cleaned = clean_segments({"text": "", "segments": [loop, real]})["segments"]
    assert cleaned[0] == {"start": 0.0, "end": 1.0, "text": "Tak,"}
    assert cleaned[1]["words"] == words[:1]


def test_clean_segments_without_segments_collapses_the_text():
    assert clean_segments({"text": " ja ja ja ja ", "segments": []})["text"] == "ja"
    assert clean_segments({"text": "kept"})["text"] == "kept"
