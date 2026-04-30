from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

from api.config import opensearch_client
from scripts.dedup.embeddings.encode import DEFAULT_ANALYZER, encode_text, load_fasttext_model, load_manifest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

TEXT_INDEX_NAME = "bec_texts"
DEFAULT_FIELD = "text_bo_embedding"
SCROLL_SIZE = 50
SCROLL_TIMEOUT = "5m"
BULK_BATCH_SIZE = 50


def _scroll_docs(index_name: str, limit: int) -> list[dict[str, Any]]:
    body: dict[str, Any] = {
        "query": {"exists": {"field": "text_bo"}},
        "_source": ["text_bo"],
    }
    response = opensearch_client.search(index=index_name, body=body, size=SCROLL_SIZE, scroll=SCROLL_TIMEOUT)
    scroll_id = response.get("_scroll_id")
    docs: list[dict[str, Any]] = []
    try:
        while True:
            hits = response.get("hits", {}).get("hits", [])
            if not hits:
                break
            docs.extend({"id": hit["_id"], **hit.get("_source", {})} for hit in hits)
            if limit > 0 and len(docs) >= limit:
                return docs[:limit]
            response = opensearch_client.scroll(scroll_id=scroll_id, scroll=SCROLL_TIMEOUT)
            scroll_id = response.get("_scroll_id")
    finally:
        if scroll_id:
            opensearch_client.clear_scroll(scroll_id=scroll_id)
    return docs


def _bulk_update(index_name: str, field: str, vectors: list[tuple[str, list[float]]], *, dry_run: bool) -> int:
    if dry_run or not vectors:
        return len(vectors)
    body: list[dict[str, Any]] = []
    for doc_id, vector in vectors:
        body.append({"update": {"_index": index_name, "_id": doc_id}})
        body.append({"doc": {field: vector}})
    response = opensearch_client.bulk(body=body, refresh=False)
    if response.get("errors"):
        error_count = sum(1 for item in response.get("items", []) if item.get("update", {}).get("error"))
        logger.warning("Bulk update had %d errors", error_count)
    return len(vectors)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill FastText embeddings into bec_texts")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--index", default=TEXT_INDEX_NAME)
    parser.add_argument("--field", default=DEFAULT_FIELD)
    parser.add_argument("--pooling", choices=["mean", "max"], default="mean")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest(args.model_path)
    analyzer = str(manifest.get("analyzer", DEFAULT_ANALYZER))
    model = load_fasttext_model(args.model_path)
    docs = _scroll_docs(args.index, args.limit)
    logger.info("Encoding %d docs with analyzer=%s", len(docs), analyzer)

    pending: list[tuple[str, list[float]]] = []
    updated = 0
    start = time.monotonic()
    for index, doc in enumerate(docs):
        vector = encode_text(
            doc.get("text_bo", ""),
            model=model,
            analyzer=analyzer,
            pooling=args.pooling,
            index=args.index,
        )
        if vector:
            pending.append((doc["id"], vector))
        if len(pending) >= BULK_BATCH_SIZE or index == len(docs) - 1:
            updated += _bulk_update(args.index, args.field, pending, dry_run=args.dry_run)
            pending = []
        if (index + 1) % 100 == 0:
            elapsed = time.monotonic() - start
            logger.info("[%d/%d] encoded %.1f docs/s", index + 1, len(docs), (index + 1) / elapsed)

    if not args.dry_run:
        opensearch_client.indices.refresh(index=args.index)
    logger.info("Backfilled %d embeddings", updated)


if __name__ == "__main__":
    main()

