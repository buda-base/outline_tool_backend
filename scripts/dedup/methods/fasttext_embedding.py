from __future__ import annotations

from pathlib import Path
from typing import Any

from api.config import opensearch_client
from scripts.dedup.embeddings.encode import DEFAULT_ANALYZER, encode_text, load_fasttext_model, load_manifest
from scripts.dedup.methods.base import BaseDedupMethod, QueryMatch, QueryScope, TextDoc
from scripts.dedup.registry import register_method
from scripts.dedup.text import cosine

TEXT_INDEX_NAME = "bec_texts"
DEFAULT_FIELD = "text_bo_embedding"


class FastTextEmbeddingMethod(BaseDedupMethod[list[float]]):
    name = "fasttext_embedding"

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        super().__init__(options)
        model_path_raw = self.options.get("model_path")
        if not model_path_raw:
            msg = "fasttext_embedding requires model_path=/path/to/model.bin"
            raise ValueError(msg)

        self.model_path = Path(str(model_path_raw))
        self.pooling = str(self.options.get("pooling", "mean"))
        self.field = str(self.options.get("field", DEFAULT_FIELD))
        self.index_name = str(self.options.get("index", TEXT_INDEX_NAME))
        self.manifest = load_manifest(self.model_path)
        self.analyzer = str(self.manifest.get("analyzer", DEFAULT_ANALYZER))
        self.model = load_fasttext_model(self.model_path)

    @property
    def supports_pair_score(self) -> bool:
        return True

    @property
    def supports_query(self) -> bool:
        return True

    def fingerprint(self, doc: TextDoc) -> list[float]:
        return encode_text(
            doc.text_bo,
            model=self.model,
            analyzer=self.analyzer,
            pooling=self.pooling,
            index=self.index_name,
        )

    def pair_score(self, fp_a: list[float], fp_b: list[float]) -> float:
        return cosine(fp_a, fp_b)

    def query(self, doc: TextDoc, *, top_k: int, scope: QueryScope) -> list[QueryMatch]:
        vector = self.fingerprint(doc)
        if not vector:
            return []

        knn_body: dict[str, Any] = {"vector": vector, "k": top_k + 1}
        if scope.kind == "closed_set":
            knn_body["filter"] = {"ids": {"values": list(scope.mw_ids)}}

        response = opensearch_client.search(
            index=scope.index,
            body={
                "size": top_k + 1,
                "_source": False,
                "query": {"knn": {self.field: knn_body}},
            },
        )
        return [
            QueryMatch(mw_id=hit["_id"], score=float(hit.get("_score", 0.0)))
            for hit in response.get("hits", {}).get("hits", [])
            if hit["_id"] != doc.mw_id
        ][:top_k]


@register_method("fasttext_embedding")
def create(options: dict[str, Any]) -> FastTextEmbeddingMethod:
    return FastTextEmbeddingMethod(options)

