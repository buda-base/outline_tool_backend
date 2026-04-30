from __future__ import annotations

from typing import Any

from opensearchpy.exceptions import ConnectionTimeout

from api.config import opensearch_client
from scripts.dedup.methods.base import BaseDedupMethod, QueryMatch, QueryScope, TextDoc
from scripts.dedup.registry import register_method

SIDECAR_INDEX = "bec_texts_minhash_3"
DEFAULT_FIELD = "text_bo.min_hash_lenient_3"
DEFAULT_BUCKET_COUNT = 512
MTERMVECTORS_BATCH_SIZE = 100
MTERMVECTORS_REQUEST_TIMEOUT = 60


def _terms_from_termvectors(doc: dict[str, Any], field: str) -> list[str]:
    terms = doc.get("term_vectors", {}).get(field, {}).get("terms", {})
    return list(terms)


class MinhashOSSidecarMethod(BaseDedupMethod[set[str]]):
    """OpenSearch MinHash query path against a sidecar index."""

    name = "minhash_os_sidecar"

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        super().__init__(options)
        self.index_name = str(self.options.get("index", SIDECAR_INDEX))
        self.field = str(self.options.get("field", DEFAULT_FIELD))
        self.bucket_count = int(self.options.get("bucket_count", DEFAULT_BUCKET_COUNT))
        self.msm_pct = float(self.options.get("msm_pct", 0.10))
        self.minimum_should_match = max(1, int(self.bucket_count * self.msm_pct))
        self.mtermvectors_batch_size = int(self.options.get("mtermvectors_batch_size", MTERMVECTORS_BATCH_SIZE))
        self._hash_cache: dict[str, list[str]] = {}

    @property
    def supports_pair_score(self) -> bool:
        return True

    @property
    def supports_query(self) -> bool:
        return True

    @property
    def requires_source_text(self) -> bool:
        return False

    def preload(self, docs: list[TextDoc]) -> None:
        missing = [doc.mw_id for doc in docs if doc.mw_id not in self._hash_cache]
        for start in range(0, len(missing), self.mtermvectors_batch_size):
            batch = missing[start : start + self.mtermvectors_batch_size]
            self._preload_batch(batch)

    def _preload_batch(self, doc_ids: list[str]) -> None:
        try:
            response = opensearch_client.mtermvectors(
                index=self.index_name,
                body={
                    "ids": doc_ids,
                    "parameters": {
                        "fields": [self.field],
                        "term_statistics": False,
                        "field_statistics": False,
                    },
                },
                request_timeout=MTERMVECTORS_REQUEST_TIMEOUT,
            )
        except ConnectionTimeout:
            if len(doc_ids) == 1:
                self._hash_cache.setdefault(doc_ids[0], [])
                return
            midpoint = len(doc_ids) // 2
            self._preload_batch(doc_ids[:midpoint])
            self._preload_batch(doc_ids[midpoint:])
            return

        for doc in response.get("docs", []):
            self._hash_cache[doc.get("_id", "")] = _terms_from_termvectors(doc, self.field)
        for doc_id in doc_ids:
            self._hash_cache.setdefault(doc_id, [])

    def _hashes(self, doc_id: str) -> list[str]:
        if doc_id in self._hash_cache:
            return self._hash_cache[doc_id]

        response = opensearch_client.termvectors(
            index=self.index_name,
            id=doc_id,
            fields=[self.field],
        )
        self._hash_cache[doc_id] = _terms_from_termvectors(response, self.field)
        return self._hash_cache[doc_id]

    def fingerprint(self, doc: TextDoc) -> set[str]:
        return set(self._hashes(doc.mw_id))

    def pair_score(self, fp_a: set[str], fp_b: set[str]) -> float:
        if not fp_a or not fp_b:
            return 0.0
        return len(fp_a & fp_b)

    def query(self, doc: TextDoc, *, top_k: int, scope: QueryScope) -> list[QueryMatch]:
        hashes = self._hashes(doc.mw_id)
        if not hashes:
            return []

        filters: list[dict[str, Any]] = []
        if scope.kind == "closed_set":
            filters.append({"terms": {"mw_id": list(scope.mw_ids)}})

        body: dict[str, Any] = {
            "size": top_k + 1,
            "_source": ["mw_id"],
            "query": {
                "bool": {
                    "filter": filters,
                    "should": [{"term": {self.field: token}} for token in hashes],
                    "minimum_should_match": self.minimum_should_match,
                    "must_not": [{"term": {"_id": doc.mw_id}}],
                }
            },
        }
        response = opensearch_client.search(index=self.index_name, body=body)
        return [
            QueryMatch(mw_id=hit.get("_source", {}).get("mw_id", hit["_id"]), score=float(hit.get("_score", 0.0)))
            for hit in response.get("hits", {}).get("hits", [])
            if hit["_id"] != doc.mw_id
        ][:top_k]


@register_method("minhash_os_sidecar")
def create(options: dict[str, Any]) -> MinhashOSSidecarMethod:
    return MinhashOSSidecarMethod(options)
