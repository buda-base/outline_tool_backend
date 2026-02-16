from typing import Annotated

from fastapi import APIRouter, Query, status

from api.exceptions import NotFoundError
from api.models import MergeRequest, PersonInput, PersonOutput
from api.services.records import create_person, get_person, merge_person, search_persons, update_person

router = APIRouter(prefix="/persons", tags=["persons"])


@router.get("/search")
async def find_person(
    author_name: Annotated[str, Query()],
) -> list[PersonOutput]:
    """Search persons by name."""
    return search_persons(author_name=author_name)


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
