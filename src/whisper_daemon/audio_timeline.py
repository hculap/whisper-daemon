"""Keep recorded audio on the same timeline as timestamped transcription chunks."""
import numpy as np


def mix_timeline(chunks, sample_rate=16000):
    if not chunks:
        return np.array([], dtype=np.float32)
    length = max(round(start * sample_rate) + len(audio) for start, audio in chunks)
    mixed = np.zeros(length, dtype=np.float32)
    counts = np.zeros(length, dtype=np.uint16)
    for start, audio in chunks:
        offset = round(start * sample_rate)
        if offset < 0:
            raise ValueError('Negative audio timestamp')
        end = offset + len(audio)
        mixed[offset:end] += audio
        counts[offset:end] += 1
    np.divide(mixed, np.maximum(counts, 1), out=mixed)
    return mixed
