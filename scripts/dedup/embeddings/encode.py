from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Protocol

from api.config import opensearch_client

TEXT_INDEX_NAME = "bec_texts"
DEFAULT_ANALYZER = "tibetan-lenient"
MAX_ANALYZE_CHARS = 15000


class VectorLike(Protocol):
    def tolist(self) -> list[float]:
        ...


class FastTextModel(Protocol):
    def get_word_vector(self, word: str) -> VectorLike:
        ...


def analyze_tokens(text: str, *, analyzer: str = DEFAULT_ANALYZER, index: str = TEXT_INDEX_NAME) -> list[str]:
    response = opensearch_client.indices.analyze(
        index=index,
        body={"analyzer": analyzer, "text": text[:MAX_ANALYZE_CHARS]},
    )
    return [token["token"] for token in response.get("tokens", [])]


def load_fasttext_model(model_path: Path) -> FastTextModel:
    try:
        import fasttext  # noqa: PLC0415
    except ImportError as exc:
        msg = "Install fasttext-wheel to use FastText embedding methods"
        raise RuntimeError(msg) from exc
    return fasttext.load_model(str(model_path))


def load_manifest(model_path: Path) -> dict[str, Any]:
    manifest_path = model_path.with_suffix(model_path.suffix + ".manifest.json")
    if not manifest_path.exists():
        return {}
    with manifest_path.open(encoding="utf-8") as fh:
        return json.load(fh)


def l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def pool_vectors(vectors: list[list[float]], *, pooling: str) -> list[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    if pooling == "max":
        return [max(vector[idx] for vector in vectors) for idx in range(dim)]
    if pooling != "mean":
        msg = f"Unsupported pooling mode: {pooling}"
        raise ValueError(msg)
    return [sum(vector[idx] for vector in vectors) / len(vectors) for idx in range(dim)]


def encode_text(
    text: str,
    *,
    model: FastTextModel,
    analyzer: str = DEFAULT_ANALYZER,
    pooling: str = "mean",
    index: str = TEXT_INDEX_NAME,
) -> list[float]:
    tokens = analyze_tokens(text, analyzer=analyzer, index=index)
    vectors = [model.get_word_vector(token).tolist() for token in tokens]
    return l2_normalize(pool_vectors(vectors, pooling=pooling))

