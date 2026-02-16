import logging
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)


def bulk_upsert_from_import(records: list[dict[str, Any]], now: str | None = None) -> dict[str, int]:
    """
    Bulk upsert records from a BDRC import using scripted_upsert.

    For each record:
    - If curation.modified == false (or doc is new): overwrite source-owned fields.
    - If curation.modified == true: update only source_meta.* + import.*, skip content fields.
    - New docs get origin="imported", record_status="active", default curation block.

    Args:
        records: list of dicts, each must have at least "id", "type", and the content fields.
        now: ISO timestamp for the import run (defaults to current time).

    Returns:
        Counts: {"updated": N, "created": N, "skipped": N}
    """
    if now is None:
        now = datetime.now(UTC).isoformat()

    # TODO: Implement the actual scripted_upsert bulk logic.
    # The Painless script from the spec should:
    #   1. Always update source_meta.updated_at and import.last_run_at
    #   2. Only overwrite content fields (pref_label_bo, alt_label_bo, etc.)
    #      when curation.modified is false
    #   3. Set import.last_result to 'updated_or_created' or 'skipped_modified'
    #
    # This requires:
    #   - Building NDJSON bulk payload with scripted_upsert
    #   - Calling opensearch_client.bulk()
    #   - Parsing the response to count results
    #   - Logging each upsert to the audit index

    logger.warning("bulk_upsert_from_import is not yet implemented — %d records ignored", len(records))
    return {"updated": 0, "created": 0, "skipped": len(records)}
