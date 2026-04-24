from typing import Annotated

from fastapi import APIRouter, Query

from api.models import Stats, VolumeBatchStatusReport
from api.services.stats import get_stats, get_volume_batch_status_report

router = APIRouter(tags=["stats"])


@router.get("/stats")
async def stats() -> Stats:
    """Get aggregated statistics for reporting."""
    return get_stats()


@router.get("/stats/volume-batches")
async def volume_batch_status_report(
    max_batches: Annotated[int, Query(description="Max distinct batch_id buckets", ge=1, le=20000)] = 5000,
) -> VolumeBatchStatusReport:
    """Count volumes per ``batch_id`` and per annotation ``status``; omits documents without a batch."""
    return VolumeBatchStatusReport.model_validate(
        get_volume_batch_status_report(max_batches=max_batches),
    )
