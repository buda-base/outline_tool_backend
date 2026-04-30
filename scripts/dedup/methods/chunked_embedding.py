from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.dedup.embeddings.chunking import equal_syllable_chunks
from scripts.dedup.embeddings.encode import DEFAULT_ANALYZER, encode_text, load_fasttext_model, load_manifest
from scripts.dedup.methods.base import BaseDedupMethod, TextDoc
from scripts.dedup.registry import register_method
from scripts.dedup.text import cosine

TEXT_INDEX_NAME = "bec_texts"


class ChunkedEmbeddingMethod(BaseDedupMethod[list[list[float]]]):
    name = "chunked_embedding"

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        super().__init__(options)
        model_path_raw = self.options.get("model_path")
        if not model_path_raw:
            msg = "chunked_embedding requires model_path=/path/to/model.bin"
            raise ValueError(msg)

        self.model_path = Path(str(model_path_raw))
        self.n_chunks = int(self.options.get("n_chunks", 10))
        self.pooling = str(self.options.get("pooling", "mean"))
        self.chunk_threshold = float(self.options.get("chunk_threshold", 0.75))
        self.index_name = str(self.options.get("index", TEXT_INDEX_NAME))
        self.manifest = load_manifest(self.model_path)
        self.analyzer = str(self.manifest.get("analyzer", DEFAULT_ANALYZER))
        self.model = load_fasttext_model(self.model_path)

    @property
    def supports_pair_score(self) -> bool:
        return True

    def fingerprint(self, doc: TextDoc) -> list[list[float]]:
        return [
            encode_text(
                chunk,
                model=self.model,
                analyzer=self.analyzer,
                pooling=self.pooling,
                index=self.index_name,
            )
            for chunk in equal_syllable_chunks(doc.text_bo, self.n_chunks)
        ]

    def _directed_fraction(self, left: list[list[float]], right: list[list[float]]) -> float:
        if not left or not right:
            return 0.0
        matching = sum(
            1
            for left_chunk in left
            if any(cosine(left_chunk, right_chunk) >= self.chunk_threshold for right_chunk in right)
        )
        return matching / len(left)

    def pair_score(self, fp_a: list[list[float]], fp_b: list[list[float]]) -> float:
        return min(self._directed_fraction(fp_a, fp_b), self._directed_fraction(fp_b, fp_a))


@register_method("chunked_embedding")
def create(options: dict[str, Any]) -> ChunkedEmbeddingMethod:
    return ChunkedEmbeddingMethod(options)

