from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status

from api.exceptions import NotFoundError
from api.models import MergeRequest, WorkInput, WorkOutput
from api.services.records import create_work, get_work, merge_work, search_works, update_work

router = APIRouter(prefix="/works", tags=["works"])


@router.get("/search")
async def find_work(
    title: Annotated[str | None, Query()] = None,
    author_name: Annotated[str | None, Query()] = None,
) -> list[WorkOutput]:
    """Search works by title and/or author name."""
    if not title and not author_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one of 'title' or 'author_name' must be provided",
        )
    return search_works(title=title, author_name=author_name)


@router.get("/{work_id}")
async def get_work_data(work_id: str) -> WorkOutput:
    """Get work data by ID."""
    work = get_work(work_id)
    if work is None:
        raise NotFoundError("Work", work_id)
    return work


@router.post("", status_code=status.HTTP_201_CREATED)
async def post_work_data(body: WorkInput) -> dict[str, str]:
    """Create a new work with a server-generated ID."""
    work = create_work(body)
    return {"id": work.id}


@router.put("/{work_id}")
async def put_work_data(work_id: str, body: WorkInput) -> WorkOutput:
    """Update an existing work (only the provided fields)."""
    return update_work(work_id, body)


@router.post("/{work_id}/merge")
async def merge_work_data(work_id: str, body: MergeRequest) -> WorkOutput:
    """Mark a work as duplicate of the canonical work."""
    return merge_work(work_id, body.canonical_id, body.modified_by)
