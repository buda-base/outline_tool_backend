from typing import Annotated, Any

from fastapi import APIRouter, Query

from api.services.audit import get_history

router = APIRouter(prefix="/audit", tags=["audit"])


@router.get("/{doc_id}")
async def get_audit_history(
    doc_id: str,
    size: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list[dict[str, Any]]:
    """Get change history for a specific document, newest first."""
    return get_history(doc_id, size=size)
