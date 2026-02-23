from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status

from api.models import PaginatedResponse, VolumeInput, VolumeOutput, VolumeStatus
from api.services.volumes import get_volume, list_volumes, update_volume

router = APIRouter(prefix="/volumes", tags=["volumes"])


@router.get("")
async def get_available_volumes(
    volume_status: Annotated[VolumeStatus | None, Query(alias="status")] = None,
    etext_source: Annotated[str | None, Query()] = None,
    rep_id: Annotated[str | None, Query()] = None,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> PaginatedResponse:
    """List volumes available for annotation, with optional filters and pagination."""
    items, total = list_volumes(
        status=volume_status.value if volume_status else None,
        etext_source=etext_source,
        rep_id=rep_id,
        offset=offset,
        limit=limit,
    )
    return PaginatedResponse(total=total, offset=offset, limit=limit, items=items)


@router.get("/{rep_id}/{vol_id}")
async def get_volume_data(rep_id: str, vol_id: str) -> VolumeOutput:
    """Get full volume data by repository ID and volume ID."""
    volume = get_volume(rep_id, vol_id)
    if volume is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Volume {rep_id}/{vol_id} not found",
        )
    return volume


@router.put("/{rep_id}/{vol_id}")
async def put_volume_data(rep_id: str, vol_id: str, body: VolumeInput) -> VolumeOutput:
    """Update an existing volume (only the provided fields)."""
    return update_volume(rep_id, vol_id, body)
