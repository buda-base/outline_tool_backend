import logging
from datetime import UTC, datetime
from typing import Any

from api.config import opensearch_client

logger = logging.getLogger(__name__)

audit_index_name = "bec_changes"


def log_event(
    doc_id: str,
    doc_type: str,
    action: str,
    actor: str,
    diff: dict[str, Any] | None = None,
    correlation_id: str | None = None,
) -> None:
    """Append an audit event to the bec_changes index."""
    body: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "actor": actor,
        "type": doc_type,
        "id": doc_id,
        "action": action,
    }
    if diff is not None:
        body["diff"] = diff
    if correlation_id is not None:
        body["correlation_id"] = correlation_id

    try:
        opensearch_client.index(index=audit_index_name, body=body, refresh=False)
    except Exception:
        logger.exception("Failed to write audit event for %s/%s", doc_type, doc_id)


def get_history(doc_id: str, size: int = 50) -> list[dict[str, Any]]:
    """Retrieve audit history for a specific document, newest first."""
    body: dict[str, Any] = {
        "query": {"term": {"id": doc_id}},
        "sort": [{"timestamp": {"order": "desc"}}],
    }
    response = opensearch_client.search(index=audit_index_name, body=body, size=size)
    return [hit["_source"] for hit in response["hits"]["hits"]]
