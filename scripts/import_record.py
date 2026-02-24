from __future__ import annotations

import logging
from typing import Any

from api.models import ImportRecord, ParsedRecord, RecordStatus, SyncCounts
from api.services.audit import log_event
from api.services.catalog_import import bulk_upsert_from_import
from api.services.os_client import get_document, update_document

logger = logging.getLogger(__name__)


def _withdraw_record(record_id: str, doc_type: str) -> None:
    """Mark an existing record as withdrawn."""
    partial: dict[str, Any] = {
        "record_status": RecordStatus.WITHDRAWN.value,
    }
    update_document(record_id, partial)
    log_event(record_id, doc_type, "withdraw", "importer")
    logger.info("Withdrew %s %s", doc_type, record_id)


def _merge_record_import(
    record_id: str,
    replacement_id: str,
    doc_type: str,
) -> None:
    """Mark an existing record as duplicate, pointing to its replacement."""
    partial: dict[str, Any] = {
        "record_status": RecordStatus.DUPLICATE.value,
        "canonical_id": replacement_id,
    }
    update_document(record_id, partial)
    log_event(
        record_id,
        doc_type,
        "merge",
        "importer",
        diff={"canonical_id": replacement_id},
    )
    logger.info("Merged %s %s → %s", doc_type, record_id, replacement_id)


def process_parsed_records(
    parsed_records: list[ParsedRecord],
    entity_scores: dict[str, float],
    now: str | None = None,
) -> SyncCounts:
    """
    Process a batch of parsed trig records according to bdrc_sync_workflow rules.

    For each record:
    - Not released + not in DB: skip
    - Not released + in DB + has replaceWith: merge into replacement
    - Not released + in DB + no replacement: withdraw
    - Released: collect for bulk upsert
    """
    counts = SyncCounts()
    records_to_upsert: list[ImportRecord] = []

    for record in parsed_records:
        if not record.is_released:
            existing = get_document(record.id)

            if existing is None:
                logger.debug("Skipping unreleased %s %s (not in DB)", record.type, record.id)
                counts.skipped += 1
                continue

            if record.replacement_id:
                _merge_record_import(record.id, record.replacement_id, record.type)
                counts.merged += 1
            else:
                _withdraw_record(record.id, record.type)
                counts.withdrawn += 1
            continue

        records_to_upsert.append(
            ImportRecord(
                id=record.id,
                type=record.type,
                pref_label_bo=record.pref_label_bo,
                alt_label_bo=record.alt_label_bo,
                authors=record.authors,
                db_score=entity_scores.get(record.id),
            )
        )

    if records_to_upsert:
        bulk_counts = bulk_upsert_from_import(records_to_upsert, now=now)
        counts.upserted = bulk_counts["created"] + bulk_counts["updated"]

    logger.info(
        "Record processing complete: %d upserted, %d merged, %d withdrawn, %d skipped",
        counts.upserted,
        counts.merged,
        counts.withdrawn,
        counts.skipped,
    )
    return counts
