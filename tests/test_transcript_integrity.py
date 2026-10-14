from concurrent.futures import Future

from whisper_daemon.__main__ import _collect_results, _merge_results
from whisper_daemon.formats import merge_chunk_results


def result(text, start, end, source="mic", words=None):
    segment = {"start": start, "end": end, "text": text}
    if words is not None:
        segment["words"] = words
    return {"text": text, "source": source, "segments": [segment], "language": "pl"}


def test_overlapping_sources_keep_both_speakers():
    merged = merge_chunk_results([
        (0, result("Local proposal", 0, 10)),
        (0, result("Remote reply", 2, 4, "browser")),
    ])
    assert merged["text"] == "Local proposal Remote reply"
    assert [s["source"] for s in merged["segments"]] == ["mic", "browser"]


def test_overlap_keeps_new_words_after_boundary():
    merged = merge_chunk_results([
        (0, result("previous shared", 0, 20)),
        (18, result("shared new decision", 0, 5, words=[
            {"word": "shared", "start": 0, "end": 1},
            {"word": "new", "start": 2, "end": 3},
            {"word": "decision", "start": 3, "end": 5},
        ])),
    ])
    assert merged["text"] == "previous shared new decision"
    assert merged["segments"][-1]["end"] == 23


def test_without_word_times_only_exact_boundary_text_is_deduplicated():
    merged = merge_chunk_results([
        (0, result("we agreed on", 0, 20)),
        (18, result("agreed on a date", 0, 5)),
    ])
    assert merged["text"] == "we agreed on a date"


def test_cli_collects_segments_instead_of_discarding_text():
    future = Future()
    future.set_result(result("recorded words", 0, 2))
    results = []
    _collect_results({future: 10.0}, results)
    merged = _merge_results(results)
    assert merged["text"] == "recorded words"
    assert merged["segments"][0]["start"] == 10


def test_legacy_text_results_are_not_silently_erased():
    assert merge_chunk_results([(0, {"text": "legacy words", "segments": []})])["text"] == "legacy words"
