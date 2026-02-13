from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status

from api.models import PersonInput, PersonOutput
from api.services.opensearch import create_person, get_person, search_persons, update_person

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
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Person '{person_id}' not found",
        )
    return person


@router.post("", status_code=status.HTTP_201_CREATED)
async def post_person_data(body: PersonInput) -> dict[str, str]:
    """Create a new person with a server-generated ID."""
    person = create_person(body)
    return {"id": person.id}


@router.put("/{person_id}")
async def put_person_data(person_id: str, body: PersonInput) -> dict[str, str]:
    """Update an existing person (only the provided fields)."""
    update_person(person_id, body)
    return {"id": person_id}
