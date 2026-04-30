from __future__ import annotations

from typing import TYPE_CHECKING

from opensearchpy.exceptions import ConnectionTimeout

import scripts.dedup.methods.minhash_os_sidecar as sidecar
from scripts.dedup.methods.base import QueryScope, TextDoc
from scripts.dedup.methods.minhash_os_sidecar import MinhashOSSidecarMethod

if TYPE_CHECKING:
    import pytest


class FakeOpenSearch:
    def __init__(self) -> None:
        self.mtermvector_calls: list[list[str]] = []
        self.termvector_calls: list[str] = []
        self.search_bodies: list[dict[str, object]] = []

    def mtermvectors(
        self,
        *,
        index: str,
        body: dict[str, object],
        request_timeout: int,
    ) -> dict[str, object]:
        del index, request_timeout
        ids = list(body["ids"])
        self.mtermvector_calls.append(ids)
        return {
            "docs": [
                {"_id": doc_id, "term_vectors": {sidecar.DEFAULT_FIELD: {"terms": {f"{doc_id}-x": {}, "shared": {}}}}}
                for doc_id in ids
            ]
        }

    def termvectors(self, *, index: str, id: str, fields: list[str]) -> dict[str, object]:  # noqa: A002
        del index, fields
        self.termvector_calls.append(id)
        return {"term_vectors": {sidecar.DEFAULT_FIELD: {"terms": {"fallback": {}, "shared": {}}}}}

    def search(self, *, index: str, body: dict[str, object]) -> dict[str, object]:
        del index
        self.search_bodies.append(body)
        return {
            "hits": {
                "hits": [
                    {"_id": "source", "_source": {"mw_id": "source"}, "_score": 10.0},
                    {"_id": "candidate", "_source": {"mw_id": "candidate"}, "_score": 3.0},
                ]
            }
        }


def test_preload_batches_and_fingerprint_uses_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = FakeOpenSearch()
    monkeypatch.setattr(sidecar, "opensearch_client", fake_client)
    method = MinhashOSSidecarMethod({"mtermvectors_batch_size": 2})
    docs = [TextDoc("a", ""), TextDoc("b", ""), TextDoc("c", "")]

    method.preload(docs)

    assert fake_client.mtermvector_calls == [["a", "b"], ["c"]]
    assert method.fingerprint(docs[0]) == {"a-x", "shared"}
    assert fake_client.termvector_calls == []


def test_preload_timeout_splits_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    class TimeoutThenSuccess(FakeOpenSearch):
        def mtermvectors(
            self,
            *,
            index: str,
            body: dict[str, object],
            request_timeout: int,
        ) -> dict[str, object]:
            ids = list(body["ids"])
            if len(ids) > 1:
                raise ConnectionTimeout("GET", "url", "timeout")
            return super().mtermvectors(index=index, body=body, request_timeout=request_timeout)

    fake_client = TimeoutThenSuccess()
    monkeypatch.setattr(sidecar, "opensearch_client", fake_client)

    MinhashOSSidecarMethod({"mtermvectors_batch_size": 2}).preload([TextDoc("a", ""), TextDoc("b", "")])

    assert fake_client.mtermvector_calls == [["a"], ["b"]]


def test_query_uses_cached_hashes_and_closed_set_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = FakeOpenSearch()
    monkeypatch.setattr(sidecar, "opensearch_client", fake_client)
    method = MinhashOSSidecarMethod({"msm_pct": 0.055})
    method._hash_cache["source"] = ["h1", "h2"]  # noqa: SLF001

    matches = method.query(
        TextDoc("source", ""),
        top_k=5,
        scope=QueryScope.closed_set({"source", "candidate"}),
    )

    assert matches == [sidecar.QueryMatch("candidate", 3.0)]
    assert fake_client.termvector_calls == []
    assert fake_client.search_bodies[0]["query"]["bool"]["minimum_should_match"] == 28
