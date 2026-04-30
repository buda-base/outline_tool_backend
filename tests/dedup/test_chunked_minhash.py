from __future__ import annotations

from typing import TYPE_CHECKING

from scripts.dedup.methods.base import TextDoc
from scripts.dedup.methods.chunked_minhash import ChunkedMinhashMethod

if TYPE_CHECKING:
    import pytest


def test_pair_score_is_symmetric_fraction_of_matching_chunks() -> None:
    method = ChunkedMinhashMethod({"chunk_threshold": 0.5})

    left = [{"a", "b"}, {"c", "d"}]
    right = [{"a", "b"}, {"x", "y"}]

    assert method.pair_score(left, right) == 0.5


def test_pair_score_returns_zero_for_empty_fingerprints() -> None:
    method = ChunkedMinhashMethod()

    assert method.pair_score([], [{"a"}]) == 0.0
    assert method.pair_score([{"a"}], []) == 0.0


def test_fingerprint_creates_one_fingerprint_per_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    method = ChunkedMinhashMethod({"n_chunks": 2, "bucket_count": 8})
    monkeypatch.setattr(method, "_tokens", lambda text: text.split())

    fingerprint = method.fingerprint(TextDoc("doc", "a་b་c་d"))

    assert len(fingerprint) == 2
    assert all(fingerprint)
