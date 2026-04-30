from __future__ import annotations

import time
from typing import Any

from api.config import opensearch_client
from scripts.dedup.methods.base import BaseDedupMethod, QueryMatch, QueryScope, TextDoc
from scripts.dedup.registry import register_method

TEXT_INDEX_NAME = "bec_texts"
DEFAULT_FIELD = "text_bo.min_hash_lenient"
DEFAULT_BUCKET_COUNT = 512


class MinhashOSQueryMethod(BaseDedupMethod[None]):
    """Production OpenSearch MinHash query path."""

    name = "minhash_os_query"

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        super().__init__(options)
        self.field = str(self.options.get("field", DEFAULT_FIELD))
        self.index_name = str(self.options.get("index", TEXT_INDEX_NAME))
        self.bucket_count = int(self.options.get("bucket_count", DEFAULT_BUCKET_COUNT))
        self.msm_pct = float(self.options.get("msm_pct", 0.10))
        self.minimum_should_match = max(1, int(self.bucket_count * self.msm_pct))

    @property
    def supports_query(self) -> bool:
        return True

    @property
    def requires_source_text(self) -> bool:
        return False

    def _hashes(self, doc_id: str) -> list[str]:
        response = opensearch_client.termvectors(
            index=self.index_name,
            id=doc_id,
            fields=[self.field],
        )
        terms = response.get("term_vectors", {}).get(self.field, {}).get("terms", {})
        return list(terms)

    def query(self, doc: TextDoc, *, top_k: int, scope: QueryScope) -> list[QueryMatch]:
        hashes = self._hashes(doc.mw_id)
        if not hashes:
            return []

        filters: list[dict[str, Any]] = []
        if scope.kind == "closed_set":
            filters.append({"ids": {"values": list(scope.mw_ids)}})

        body: dict[str, Any] = {
            "size": top_k + 1,
            "_source": False,
            "query": {
                "bool": {
                    "filter": filters,
                    "should": [{"term": {self.field: token}} for token in hashes],
                    "minimum_should_match": self.minimum_should_match,
                    "must_not": [{"term": {"_id": doc.mw_id}}],
                }
            },
        }

        start = time.monotonic()
        response = opensearch_client.search(index=scope.index, body=body)
        elapsed_ms = (time.monotonic() - start) * 1000
        del elapsed_ms

        return [
            QueryMatch(mw_id=hit["_id"], score=float(hit.get("_score", 0.0)))
            for hit in response.get("hits", {}).get("hits", [])
            if hit["_id"] != doc.mw_id
        ][:top_k]


@register_method("minhash_os_query")
def create(options: dict[str, Any]) -> MinhashOSQueryMethod:
    return MinhashOSQueryMethod(options)

