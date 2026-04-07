from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status

from api.exceptions import NotFoundError
from api.models import MergeRequest, Origin, PersonInput, PersonOutput, PersonsPaginatedResponse, RecordStatus
from api.services.records import create_person, delete_person, get_person, list_persons, merge_person, search_persons, update_person

router = APIRouter(prefix="/persons", tags=["persons"])


@router.get("")
async def list_person_records(
    modified_by: Annotated[str | None, Query()] = None,
    pref_label_bo: Annotated[str | None, Query()] = None,
    record_origin: Annotated[Origin | None, Query()] = None,
    record_status: Annotated[RecordStatus | None, Query()] = None,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> PersonsPaginatedResponse:
    """List persons with optional catalog filters and pagination (same filters as ``/persons/search``)."""
    items, total = list_persons(
        modified_by=modified_by,
        pref_label_bo=pref_label_bo,
        record_origin=record_origin,
        record_status=record_status,
        offset=offset,
        limit=limit,
    )
    return PersonsPaginatedResponse(total=total, offset=offset, limit=limit, items=items)


@router.get("/search")
async def find_person(
    author_name: Annotated[str | None, Query()] = None,
    modified_by: Annotated[str | None, Query()] = None,
    pref_label_bo: Annotated[str | None, Query()] = None,
    record_origin: Annotated[Origin | None, Query()] = None,
    record_status: Annotated[RecordStatus | None, Query()] = None,
) -> list[PersonOutput]:
    """Search persons by name, with optional catalog filters."""
    has_text = bool(author_name and author_name.strip())
    has_filters = any(
        v is not None for v in (modified_by, pref_label_bo, record_origin, record_status)
    )
    if not has_text and not has_filters:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide author_name or at least one filter "
            "(modified_by, pref_label_bo, record_origin, record_status)",
        )
    return search_persons(
        author_name=author_name,
        modified_by=modified_by,
        pref_label_bo=pref_label_bo,
        record_origin=record_origin,
        record_status=record_status,
    )


@router.get("/{person_id}")
async def get_person_data(person_id: str) -> PersonOutput:
    """Get person data by ID."""
    person = get_person(person_id)
    if person is None:
        raise NotFoundError("Person", person_id)
    return person


@router.post("", status_code=status.HTTP_201_CREATED)
async def post_person_data(body: PersonInput) -> dict[str, str]:
    """Create a new person with a server-generated ID."""
    person = create_person(body)
    return {"id": person.id}


@router.put("/{person_id}")
async def put_person_data(person_id: str, body: PersonInput) -> PersonOutput:
    """Update an existing person (only the provided fields)."""
    return update_person(person_id, body)


@router.post("/{person_id}/merge")
async def merge_person_data(person_id: str, body: MergeRequest) -> PersonOutput:
    """Mark a person as duplicate of the canonical person."""
    return merge_person(person_id, body.canonical_id, body.modified_by)


@router.delete("/{person_id}")
async def delete_person_data(person_id: str, modified_by: str) -> PersonOutput:
    """Soft-delete a locally created person (prefix 'P1BC') that is not listed as author in any work."""
    return delete_person(person_id, modified_by)
