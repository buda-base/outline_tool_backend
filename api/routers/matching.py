from fastapi import APIRouter, HTTPException, status

from api.models import MatchCandidate, MatchRequest
from api.services.matching import find_matching_works, find_matching_works_by_volume_ref

router = APIRouter(prefix="/matching", tags=["matching"])


@router.post("/find-work")
async def find_work(body: MatchRequest) -> list[MatchCandidate]:
    """Find candidate works matching the given text content.

    Accepts either raw text (text_bo) or a volume reference
    (volume_id + cstart + cend) to extract text from.
    """
    if body.text_bo is not None:
        try:
            return find_matching_works(body.text_bo)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            ) from e

    # model_validator guarantees these are not None when text_bo is None
    if body.volume_id is None or body.cstart is None or body.cend is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="volume_id, cstart, and cend are required when text_bo is not provided",
        )
    try:
        return find_matching_works_by_volume_ref(
            body.volume_id,
            body.cstart,
            body.cend,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
