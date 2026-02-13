import logging
from typing import Any

from fastapi import APIRouter, BackgroundTasks

from api.models import ImportOCRRequest
from api.services.ocr_import import import_ocr_from_s3

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/import", tags=["import"])


def _import_ocr_volume_task(
    w_id: str,
    i_id: str,
    i_version: str,
    source: str,
) -> None:
    """Background task that downloads a parquet from S3 and imports the OCR volume."""
    try:
        doc_id = import_ocr_from_s3(
            w_id=w_id,
            i_id=i_id,
            i_version=i_version,
            source=source,
        )
        logger.info("Import completed: %s", doc_id)
    except Exception:
        logger.exception("Import failed for %s/%s", w_id, i_id)


@router.post("/ocr-volume")
async def import_ocr_volume(body: ImportOCRRequest, background_tasks: BackgroundTasks) -> dict[str, Any]:
    """Queue an OCR volume import — downloads parquet from S3 and indexes it."""
    background_tasks.add_task(
        _import_ocr_volume_task,
        w_id=body.w_id,
        i_id=body.i_id,
        i_version=body.i_version,
        source=body.source,
    )
    return {
        "status": "accepted",
        "message": f"Import queued for {body.w_id}/{body.i_id}",
    }


@router.post("/sync-catalog")
async def sync_catalog() -> dict[str, Any]:
    """Import persons and works from the BDRC catalog."""
    # TODO: implement BDRC catalog sync
    return {
        "status": "not_implemented",
        "message": "Catalog sync is not yet implemented",
    }
