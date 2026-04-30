from __future__ import annotations

from scripts.dedup.text import tibetan_syllables


def equal_syllable_chunks(text: str, n_chunks: int) -> list[str]:
    """Split text into N near-equal syllable chunks."""
    if n_chunks <= 0:
        msg = "n_chunks must be positive"
        raise ValueError(msg)

    syllables = tibetan_syllables(text)
    if not syllables:
        return []

    chunk_count = min(n_chunks, len(syllables))
    chunks: list[str] = []
    for idx in range(chunk_count):
        start = round(idx * len(syllables) / chunk_count)
        end = round((idx + 1) * len(syllables) / chunk_count)
        if start < end:
            chunks.append("་".join(syllables[start:end]))
    return chunks

