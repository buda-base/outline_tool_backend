from fastapi import APIRouter

from api.models import Stats
from api.services.stats import get_stats

router = APIRouter(tags=["stats"])


@router.get("/stats")
async def stats() -> Stats:
    """Get aggregated statistics for reporting."""
    return get_stats()
