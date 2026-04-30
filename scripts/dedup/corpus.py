from __future__ import annotations

import csv
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from opensearchpy.exceptions import ConnectionTimeout

from api.config import opensearch_client
from scripts.dedup.methods.base import TextDoc

logger = logging.getLogger(__name__)

TEXT_INDEX_NAME = "bec_texts"
MGET_BATCH_SIZE = 5
MGET_ID_ONLY_BATCH_SIZE = 100
MGET_REQUEST_TIMEOUT = 60


@dataclass(frozen=True)
class GroundTruthRow:
    mw_id: str
    d_id: str
    rkts_id: str | None = None
    nlm_id: str | None = None


@dataclass(frozen=True)
class CorpusConfig:
    csv_path: Path
    filter_in_index: bool = True
    allowlist_path: Path | None = None
    denylist_path: Path | None = None
    index_name: str = TEXT_INDEX_NAME
    mw_id_subset: frozenset[str] | None = None
    load_source_text: bool = True


def _load_id_file(path: Path | None) -> set[str]:
    if path is None:
        return set()
    with path.open(encoding="utf-8") as fh:
        return {line.strip() for line in fh if line.strip() and not line.startswith("#")}


def load_ground_truth_rows(config: CorpusConfig) -> dict[str, GroundTruthRow]:
    allowlist = _load_id_file(config.allowlist_path)
    denylist = _load_id_file(config.denylist_path)

    rows: dict[str, GroundTruthRow] = {}
    with config.csv_path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            mw_id = (row.get("mw_id") or "").strip()
            d_id = (row.get("d_id") or "").strip()
            if not mw_id or not d_id:
                continue
            if allowlist and mw_id not in allowlist:
                continue
            if mw_id in denylist:
                continue
            if config.mw_id_subset is not None and mw_id not in config.mw_id_subset:
                continue
            rows[mw_id] = GroundTruthRow(
                mw_id=mw_id,
                d_id=d_id,
                rkts_id=(row.get("rkts_id") or "").strip() or None,
                nlm_id=(row.get("nlm_id") or "").strip() or None,
            )
    return rows


def _mget_sources(index_name: str, mw_ids: list[str], *, load_source_text: bool) -> dict[str, dict[str, Any]]:
    sources: dict[str, dict[str, Any]] = {}
    batch_size = MGET_BATCH_SIZE if load_source_text else MGET_ID_ONLY_BATCH_SIZE
    for start in range(0, len(mw_ids), batch_size):
        batch = mw_ids[start : start + batch_size]
        logger.info("Fetching source docs %d-%d/%d from %s", start + 1, start + len(batch), len(mw_ids), index_name)
        sources.update(_mget_batch(index_name, batch, load_source_text=load_source_text))
    return sources


def _mget_batch(index_name: str, doc_ids: list[str], *, load_source_text: bool) -> dict[str, dict[str, Any]]:
    source_fields = (
        [
            "mw_id",
            "text_bo",
            "title_bo",
            "wa_id_orig",
            "etext_source",
            "text_length",
        ]
        if load_source_text
        else ["mw_id", "wa_id_orig"]
    )
    try:
        response = opensearch_client.mget(
            index=index_name,
            body={"ids": doc_ids},
            _source=source_fields,
            request_timeout=MGET_REQUEST_TIMEOUT,
        )
    except ConnectionTimeout:
        if len(doc_ids) == 1:
            logger.warning("Timed out fetching %s from %s; skipping", doc_ids[0], index_name)
            return {}
        logger.warning("Timed out fetching %d docs; retrying individually", len(doc_ids))
        result: dict[str, dict[str, Any]] = {}
        for doc_id in doc_ids:
            result.update(_mget_batch(index_name, [doc_id], load_source_text=load_source_text))
        return result

    return {
        doc["_id"]: doc.get("_source", {})
        for doc in response.get("docs", [])
        if doc.get("found")
    }


def load_corpus(config: CorpusConfig) -> tuple[list[TextDoc], dict[str, GroundTruthRow]]:
    """Load benchmark docs from CSV and optionally require presence in bec_texts."""
    rows = load_ground_truth_rows(config)
    sources = (
        _mget_sources(config.index_name, list(rows), load_source_text=config.load_source_text)
        if config.filter_in_index
        else {}
    )

    docs: list[TextDoc] = []
    for mw_id, row in rows.items():
        source = sources.get(mw_id)
        if config.filter_in_index and source is None:
            continue
        source = source or {}
        text = source.get("text_bo", "")
        docs.append(
            TextDoc(
                mw_id=mw_id,
                text_bo=text,
                d_id=row.d_id,
                rkts_id=row.rkts_id,
                wa_id_orig=source.get("wa_id_orig"),
                title_bo=source.get("title_bo"),
                etext_source=source.get("etext_source"),
                text_length=source.get("text_length") or len(text),
            )
        )

    if config.filter_in_index:
        logger.info("Loaded %d/%d CSV rows present in %s", len(docs), len(rows), config.index_name)
    else:
        logger.info("Loaded %d CSV rows without index filtering", len(docs))
    return docs, rows


def groups_by_field(docs: list[TextDoc], field_name: str) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for doc in docs:
        value = getattr(doc, field_name)
        if value:
            grouped[str(value)].add(doc.mw_id)
    return {key: value for key, value in grouped.items() if len(value) >= 2}


def mw_id_subset_for_groups(
    config: CorpusConfig,
    *,
    positive_field: str,
    limit_groups: int,
) -> frozenset[str] | None:
    if limit_groups <= 0:
        return config.mw_id_subset

    grouped: dict[str, set[str]] = {}
    for mw_id, row in load_ground_truth_rows(config).items():
        value = getattr(row, positive_field, None)
        if value:
            grouped.setdefault(str(value), set()).add(mw_id)

    return frozenset(
        mw_id
        for group_ids in list(grouped.values())[:limit_groups]
        for mw_id in group_ids
    )


def positive_pairs(groups: dict[str, set[str]]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for group_ids in groups.values():
        sorted_ids = sorted(group_ids)
        for left_idx, left_id in enumerate(sorted_ids):
            for right_id in sorted_ids[left_idx + 1 :]:
                pairs.add((left_id, right_id))
    return pairs

