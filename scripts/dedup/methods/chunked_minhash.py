from __future__ import annotations

from typing import Any

from api.config import opensearch_client
from scripts.dedup.embeddings.chunking import equal_syllable_chunks
from scripts.dedup.methods.base import BaseDedupMethod, TextDoc
from scripts.dedup.registry import register_method
from scripts.dedup.text import minhash_values, shingles_from_tokens, tibetan_syllables

TEXT_INDEX_NAME = "bec_texts"
MAX_ANALYZE_CHARS = 15000


def _meets_jaccard_threshold(left: set[str], right: set[str], threshold: float) -> bool:
    if not left or not right:
        return False
    intersection_size = len(left & right)
    union_size = len(left) + len(right) - intersection_size
    return union_size > 0 and intersection_size / union_size >= threshold


class ChunkedMinhashMethod(BaseDedupMethod[list[set[str]]]):
    name = "chunked_minhash"

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        super().__init__(options)
        self.n_chunks = int(self.options.get("n_chunks", 10))
        self.shingle_size = int(self.options.get("shingle_size", 1))
        self.bucket_count = int(self.options.get("bucket_count", 256))
        self.chunk_threshold = float(self.options.get("chunk_threshold", 0.6))
        self.analyzer = str(self.options.get("analyzer", "raw"))
        self.index_name = str(self.options.get("index", TEXT_INDEX_NAME))

    @property
    def supports_pair_score(self) -> bool:
        return True

    def _tokens(self, text: str) -> list[str]:
        if self.analyzer == "raw":
            return tibetan_syllables(text)
        response = opensearch_client.indices.analyze(
            index=self.index_name,
            body={"analyzer": self.analyzer, "text": text[:MAX_ANALYZE_CHARS]},
        )
        return [token["token"] for token in response.get("tokens", [])]

    def fingerprint(self, doc: TextDoc) -> list[set[str]]:
        return [
            {str(value) for value in minhash_values(shingles, num_perm=self.bucket_count)}
            for chunk in equal_syllable_chunks(doc.text_bo, self.n_chunks)
            if (shingles := shingles_from_tokens(self._tokens(chunk), self.shingle_size))
        ]

    def _directed_fraction(self, left: list[set[str]], right: list[set[str]]) -> float:
        if not left or not right:
            return 0.0
        matching = sum(
            1
            for left_chunk in left
            if any(
                _meets_jaccard_threshold(left_chunk, right_chunk, self.chunk_threshold)
                for right_chunk in right
            )
        )
        return matching / len(left)

    def pair_score(self, fp_a: list[set[str]], fp_b: list[set[str]]) -> float:
        return min(self._directed_fraction(fp_a, fp_b), self._directed_fraction(fp_b, fp_a))


@register_method("chunked_minhash")
def create(options: dict[str, Any]) -> ChunkedMinhashMethod:
    return ChunkedMinhashMethod(options)

