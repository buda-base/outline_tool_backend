import json
from typing import Any

from opensearchpy.exceptions import NotFoundError as OSNotFoundError

from api.config import index_name, opensearch_client


def index_document(doc_id: str, body: dict[str, Any], routing: str | None = None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "index": index_name,
        "id": doc_id,
        "body": body,
        "refresh": True,
    }
    if routing is not None:
        kwargs["routing"] = routing
    return opensearch_client.index(**kwargs)


def get_document(doc_id: str, routing: str | None = None) -> dict[str, Any] | None:
    kwargs: dict[str, Any] = {
        "index": index_name,
        "id": doc_id,
    }
    if routing is not None:
        kwargs["routing"] = routing
    try:
        response = opensearch_client.get(**kwargs)
        return response["_source"]
    except OSNotFoundError:
        return None


def update_document(doc_id: str, partial_body: dict[str, Any], routing: str | None = None) -> dict[str, Any]:
    """Partial update of a document (only the given fields)."""
    kwargs: dict[str, Any] = {
        "index": index_name,
        "id": doc_id,
        "body": {"doc": partial_body},
        "refresh": True,
    }
    if routing is not None:
        kwargs["routing"] = routing
    return opensearch_client.update(**kwargs)


def search(
    body: dict[str, Any],
    size: int = 50,
    offset: int = 0,
    source_excludes: list[str] | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"index": index_name, "body": body, "size": size, "from_": offset}
    if source_excludes:
        kwargs["_source_excludes"] = source_excludes
    return opensearch_client.search(**kwargs)


def bulk_operation(body: list[dict[str, Any]], *, refresh: bool = False) -> dict[str, Any]:
    ndjson = "\n".join(json.dumps(line) for line in body) + "\n"
    return opensearch_client.bulk(body=ndjson, index=index_name, refresh=refresh)


def refresh_index() -> None:
    opensearch_client.indices.refresh(index=index_name)


def extract_hits(response: dict[str, Any]) -> list[dict[str, Any]]:
    return [{**hit["_source"], "id": hit["_id"]} for hit in response["hits"]["hits"]]
