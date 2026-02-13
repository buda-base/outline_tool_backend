import logging
from datetime import UTC, datetime
from typing import Any

from opensearchpy.exceptions import NotFoundError

from api.config import index_name, opensearch_client
from api.models import (
    DocumentType,
    PersonInput,
    PersonOutput,
    VolumeInput,
    VolumeOutput,
    WorkInput,
    WorkOutput,
    generate_id,
)
from query_builder import build_search_query

logger = logging.getLogger(__name__)


def _index_document(doc_id: str, body: dict[str, Any], routing: str | None = None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "index": index_name,
        "id": doc_id,
        "body": body,
        "refresh": True,
    }
    if routing is not None:
        kwargs["routing"] = routing
    return opensearch_client.index(**kwargs)


def _get_document(doc_id: str, routing: str | None = None) -> dict[str, Any] | None:
    kwargs: dict[str, Any] = {
        "index": index_name,
        "id": doc_id,
    }
    if routing is not None:
        kwargs["routing"] = routing
    try:
        response = opensearch_client.get(**kwargs)
        return response["_source"]
    except NotFoundError:
        return None


def _update_document(doc_id: str, partial_body: dict[str, Any], routing: str | None = None) -> dict[str, Any]:
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


def _search(
    body: dict[str, Any],
    size: int = 50,
    offset: int = 0,
    source_excludes: list[str] | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"index": index_name, "body": body, "size": size, "from_": offset}
    if source_excludes:
        kwargs["_source_excludes"] = source_excludes
    return opensearch_client.search(**kwargs)


def _extract_hits(response: dict[str, Any]) -> list[dict[str, Any]]:
    return [{**hit["_source"], "id": hit["_id"]} for hit in response["hits"]["hits"]]


def _volume_doc_id(w_id: str, i_id: str) -> str:
    return f"{w_id}_{i_id}"


def list_volumes(
    status: str | None = None,
    source: str | None = None,
    w_id: str | None = None,
    offset: int = 0,
    limit: int = 50,
) -> tuple[list[VolumeOutput], int]:
    filters: list[dict[str, Any]] = [
        {"term": {"type": DocumentType.VOLUME_ETEXT.value}},
    ]
    if status is not None:
        filters.append({"term": {"status": status}})
    if source is not None:
        filters.append({"term": {"source": source}})
    if w_id is not None:
        filters.append({"term": {"w_id": w_id}})

    body: dict[str, Any] = {"query": {"bool": {"filter": filters}}}
    response = _search(
        body,
        size=limit,
        offset=offset,
        source_excludes=["chunks", "pages", "segments"],
    )
    total: int = response["hits"]["total"]["value"]
    items = [VolumeOutput.model_validate(h) for h in _extract_hits(response)]
    return items, total


def get_volume(w_id: str, i_id: str) -> VolumeOutput | None:
    doc_id = _volume_doc_id(w_id, i_id)
    source = _get_document(doc_id)
    if source is None:
        return None
    return VolumeOutput.model_validate({**source, "id": doc_id, "w_id": w_id, "i_id": i_id})


def create_volume(w_id: str, i_id: str, data: VolumeInput) -> VolumeOutput:
    """Create a new volume document."""
    doc_id = _volume_doc_id(w_id, i_id)
    now = datetime.now(UTC).isoformat()
    body = {
        **data.model_dump(),
        "type": DocumentType.VOLUME_ETEXT.value,
        "w_id": w_id,
        "i_id": i_id,
        "first_imported_at": now,
        "last_updated_at": now,
        "join_field": {"name": "instance"},
    }
    _index_document(doc_id, body)
    return VolumeOutput.model_validate({**body, "id": doc_id})


def update_volume(w_id: str, i_id: str, data: VolumeInput) -> dict[str, Any]:
    """Partial update of an existing volume (only client-sent fields)."""
    doc_id = _volume_doc_id(w_id, i_id)
    partial = data.model_dump(exclude_unset=True)
    partial["last_updated_at"] = datetime.now(UTC).isoformat()
    return _update_document(doc_id, partial)


def create_work(data: WorkInput) -> WorkOutput:
    """Create a new work document with a generated ID."""
    work_id = generate_id("W")
    body = {**data.model_dump(), "type": DocumentType.WORK.value}
    _index_document(work_id, body)
    return WorkOutput.model_validate({**body, "id": work_id})


def update_work(work_id: str, data: WorkInput) -> dict[str, Any]:
    """Partial update of an existing work (only client-sent fields)."""
    return _update_document(work_id, data.model_dump(exclude_unset=True))


def get_work(work_id: str) -> WorkOutput | None:
    source = _get_document(work_id)
    if source is None:
        return None
    return WorkOutput.model_validate({**source, "id": work_id})


def search_works(title: str | None = None, author_name: str | None = None, size: int = 20) -> list[WorkOutput]:
    type_filter: list[dict[str, Any]] = [
        {"term": {"type": DocumentType.WORK.value}},
    ]
    search_text_parts: list[str] = []
    if title:
        search_text_parts.append(title)
    if author_name:
        search_text_parts.append(author_name)

    if not search_text_parts:
        body: dict[str, Any] = {"query": {"bool": {"filter": type_filter}}}
    else:
        body = build_search_query(
            {
                "query": " ".join(search_text_parts),
                "filter": type_filter,
            }
        )

    response = _search(body, size=size)
    return [WorkOutput.model_validate(h) for h in _extract_hits(response)]


def create_person(data: PersonInput) -> PersonOutput:
    """Create a new person document with a generated ID."""
    person_id = generate_id("P")
    body = {**data.model_dump(), "type": DocumentType.PERSON.value}
    _index_document(person_id, body)
    return PersonOutput.model_validate({**body, "id": person_id})


def update_person(person_id: str, data: PersonInput) -> dict[str, Any]:
    """Partial update of an existing person (only client-sent fields)."""
    return _update_document(person_id, data.model_dump(exclude_unset=True))


def get_person(person_id: str) -> PersonOutput | None:
    source = _get_document(person_id)
    if source is None:
        return None
    return PersonOutput.model_validate({**source, "id": person_id})


def search_persons(author_name: str, size: int = 20) -> list[PersonOutput]:
    filters: list[dict[str, Any]] = [
        {"term": {"type": DocumentType.PERSON.value}},
    ]
    body = build_search_query({"query": author_name, "filter": filters})
    response = _search(body, size=size)
    return [PersonOutput.model_validate(h) for h in _extract_hits(response)]


def get_stats() -> dict[str, Any]:
    body: dict[str, Any] = {
        "size": 0,
        "aggs": {
            "by_type": {
                "terms": {"field": "type", "size": 10},
                "aggs": {
                    "by_status": {
                        "terms": {"field": "status", "size": 10},
                    },
                    "total_segments": {
                        "nested": {"path": "segments"},
                        "aggs": {
                            "count": {"value_count": {"field": "segments.cstart"}},
                        },
                    },
                },
            },
        },
    }
    return _search(body, size=0)
