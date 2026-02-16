from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status

from api.models import PaginatedResponse, VolumeInput, VolumeOutput, VolumeStatus
from api.services.volumes import get_volume, list_volumes, update_volume

router = APIRouter(prefix="/volumes", tags=["volumes"])


@router.get("")
async def get_available_volumes(
    volume_status: Annotated[VolumeStatus | None, Query(alias="status")] = None,
    etext_source: Annotated[str | None, Query()] = None,
    w_id: Annotated[str | None, Query()] = None,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> PaginatedResponse:
    """List volumes available for annotation, with optional filters and pagination."""
    items, total = list_volumes(
        status=volume_status.value if volume_status else None,
        etext_source=etext_source,
        w_id=w_id,
        offset=offset,
        limit=limit,
    )
    return PaginatedResponse(total=total, offset=offset, limit=limit, items=items)


@router.get("/{w_id}/{i_id}")
async def get_volume_data(w_id: str, i_id: str) -> VolumeOutput:
    """Get full volume data by work ID and image instance ID."""
    volume = get_volume(w_id, i_id)
    if volume is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Volume {w_id}/{i_id} not found",
        )
    return volume


@router.put("/{w_id}/{i_id}")
async def put_volume_data(w_id: str, i_id: str, body: VolumeInput) -> VolumeOutput:
    """Update an existing volume (only the provided fields)."""
    return update_volume(w_id, i_id, body)
