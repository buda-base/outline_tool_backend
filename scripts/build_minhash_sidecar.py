from __future__ import annotations

import argparse
import contextlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from api.config import opensearch_client
from scripts.dedup.corpus import CorpusConfig, groups_by_field, load_corpus, mw_id_subset_for_groups

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SOURCE_INDEX = "bec_texts"
SIDECAR_INDEX = "bec_texts_minhash_3"
MAPPING_PATH = Path("doc/mappings_bec_texts_minhash_3.json")
DEFAULT_CSV = Path("bdrc_data/nlm_merged.csv")
SCROLL_SIZE = 100
SCROLL_TIMEOUT = "5m"
BULK_BATCH_SIZE = 100


def _strip_surrogates(text: str) -> str:
    return "".join(char for char in text if not 0xD800 <= ord(char) <= 0xDFFF)


def _load_mapping(mapping_path: Path) -> dict[str, Any]:
    with mapping_path.open(encoding="utf-8") as fh:
        return json.load(fh)


def ensure_sidecar_index(index_name: str, mapping_path: Path, *, recreate: bool) -> None:
    exists = opensearch_client.indices.exists(index=index_name)
    if exists and recreate:
        logger.warning("Deleting existing sidecar index %s", index_name)
        opensearch_client.indices.delete(index=index_name)
        exists = False
    if exists:
        logger.info("Sidecar index %s already exists", index_name)
        return
    opensearch_client.indices.create(index=index_name, body=_load_mapping(mapping_path))
    logger.info("Created sidecar index %s", index_name)


def _docs_from_csv_subset(
    csv_path: Path,
    positive_field: str,
    limit_groups: int,
) -> list[dict[str, Any]]:
    subset = mw_id_subset_for_groups(
        CorpusConfig(csv_path=csv_path, filter_in_index=False),
        positive_field=positive_field,
        limit_groups=limit_groups,
    )
    docs, _rows = load_corpus(
        CorpusConfig(
            csv_path=csv_path,
            filter_in_index=True,
            mw_id_subset=subset,
        )
    )
    groups = groups_by_field(docs, positive_field)
    logger.info(
        "Loaded CSV subset for sidecar: %d docs / %d positive groups",
        len(docs),
        len(groups),
    )
    return [
        {"id": doc.mw_id, "text_bo": doc.text_bo, "text_length": doc.text_length}
        for doc in docs
    ]


def _docs_from_source_scroll(source_index: str, limit: int) -> list[dict[str, Any]]:
    body: dict[str, Any] = {
        "query": {"exists": {"field": "text_bo"}},
        "_source": ["text_bo", "text_length"],
    }
    response = opensearch_client.search(index=source_index, body=body, size=SCROLL_SIZE, scroll=SCROLL_TIMEOUT)
    scroll_id = response.get("_scroll_id")
    docs: list[dict[str, Any]] = []
    try:
        while True:
            hits = response.get("hits", {}).get("hits", [])
            if not hits:
                break
            docs.extend(
                {
                    "id": hit["_id"],
                    "text_bo": hit.get("_source", {}).get("text_bo", ""),
                    "text_length": hit.get("_source", {}).get("text_length", 0),
                }
                for hit in hits
            )
            if limit > 0 and len(docs) >= limit:
                return docs[:limit]
            response = opensearch_client.scroll(scroll_id=scroll_id, scroll=SCROLL_TIMEOUT)
            scroll_id = response.get("_scroll_id")
    finally:
        if scroll_id:
            with contextlib.suppress(Exception):
                opensearch_client.clear_scroll(scroll_id=scroll_id)
    return docs


def _bulk_index(
    *,
    docs: list[dict[str, Any]],
    sidecar_index: str,
    source_index: str,
    dry_run: bool,
) -> int:
    if not docs:
        return 0
    synced_at = datetime.now(UTC).isoformat()
    total = 0
    for start in range(0, len(docs), BULK_BATCH_SIZE):
        batch = docs[start : start + BULK_BATCH_SIZE]
        body: list[dict[str, Any]] = []
        for doc in batch:
            text = _strip_surrogates(doc.get("text_bo", ""))
            body.append({"index": {"_index": sidecar_index, "_id": doc["id"]}})
            body.append(
                {
                    "mw_id": doc["id"],
                    "source_index": source_index,
                    "source_synced_at": synced_at,
                    "text_bo": text,
                    "text_length": doc.get("text_length") or len(text),
                }
            )
        if not dry_run:
            response = opensearch_client.bulk(body=body, refresh=False)
            if response.get("errors"):
                error_count = sum(1 for item in response.get("items", []) if item.get("index", {}).get("error"))
                logger.warning("Bulk index had %d errors", error_count)
        total += len(batch)
        logger.info("Indexed %d/%d sidecar docs", total, len(docs))
    if not dry_run:
        opensearch_client.indices.refresh(index=sidecar_index)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 3-gram MinHash sidecar index from bec_texts")
    parser.add_argument("--source-index", default=SOURCE_INDEX)
    parser.add_argument("--sidecar-index", default=SIDECAR_INDEX)
    parser.add_argument("--mapping", type=Path, default=MAPPING_PATH)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--from-csv", action="store_true", help="Populate only rows from the benchmark CSV")
    parser.add_argument("--positive-field", choices=["d_id", "rkts_id", "wa_id_orig"], default="d_id")
    parser.add_argument("--limit-groups", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="Limit docs when scrolling all source docs")
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.dry_run:
        ensure_sidecar_index(args.sidecar_index, args.mapping, recreate=args.recreate)

    if args.from_csv:
        docs = _docs_from_csv_subset(args.csv, args.positive_field, args.limit_groups)
    else:
        docs = _docs_from_source_scroll(args.source_index, args.limit)

    count = _bulk_index(
        docs=docs,
        sidecar_index=args.sidecar_index,
        source_index=args.source_index,
        dry_run=args.dry_run,
    )
    logger.info("%s %d docs into %s", "Would index" if args.dry_run else "Indexed", count, args.sidecar_index)


if __name__ == "__main__":
    main()
