"""Tests for Whisper hallucination-loop collapsing."""

import pytest

from whisper_daemon.text_cleanup import collapse_repetitions


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
    ],
)
def test_preserves_real_speech(text):
    assert collapse_repetitions(text) == text


def test_empty_and_whitespace():
    assert collapse_repetitions("") == ""
    assert collapse_repetitions("   ") == ""
