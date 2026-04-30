from __future__ import annotations

from typing import Any

from api.config import opensearch_client
from scripts.dedup.methods.base import BaseDedupMethod, TextDoc
from scripts.dedup.registry import register_method
from scripts.dedup.text import jaccard, minhash_values, shingles_from_tokens

TEXT_INDEX_NAME = "bec_texts"
MAX_ANALYZE_CHARS = 15000


class MinhashOSJaccardMethod(BaseDedupMethod[set[str]]):
    """Offline MinHash proxy using OpenSearch analyzer tokens.

    This is deliberately not the production query path. It lets us benchmark
    analyzer/shingle-size options without reindexing.
    """

    name = "minhash_os_jaccard"

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        super().__init__(options)
        self.index_name = str(self.options.get("index", TEXT_INDEX_NAME))
        self.analyzer = str(self.options.get("analyzer", "tibetan-lenient"))
        self.shingle_size = int(self.options.get("shingle_size", 1))
        self.bucket_count = int(self.options.get("bucket_count", 512))

    @property
    def supports_pair_score(self) -> bool:
        return True

    def _tokens(self, text: str) -> list[str]:
        response = opensearch_client.indices.analyze(
            index=self.index_name,
            body={"analyzer": self.analyzer, "text": text[:MAX_ANALYZE_CHARS]},
        )
        return [token["token"] for token in response.get("tokens", [])]

    def fingerprint(self, doc: TextDoc) -> set[str]:
        tokens = self._tokens(doc.text_bo)
        shingles = shingles_from_tokens(tokens, self.shingle_size)
        # Use regular MinHash slots as a local proxy for OpenSearch bucket_count.
        return {str(value) for value in minhash_values(shingles, num_perm=self.bucket_count)}

    def pair_score(self, fp_a: set[str], fp_b: set[str]) -> float:
        return jaccard(fp_a, fp_b)


@register_method("minhash_os_jaccard")
def create(options: dict[str, Any]) -> MinhashOSJaccardMethod:
    return MinhashOSJaccardMethod(options)

