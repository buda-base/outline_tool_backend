import logging
from typing import Annotated, Any

from fastapi import APIRouter, BackgroundTasks, Query

from api.models import ImportOCRRequest
from api.services.ocr_import import import_ocr_from_s3
from scripts.entity_scores import load_entity_scores
from scripts.sync_bdrc import sync_repo

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/import", tags=["import"])


def _import_ocr_volume_task(
    rep_id: str,
    vol_id: str,
    vol_version: str,
    etext_source: str,
) -> None:
    """Background task that downloads a parquet from S3 and imports the OCR volume."""
    logger.info("Starting import for %s/%s (version=%s, etext_source=%s)", rep_id, vol_id, vol_version, etext_source)
    try:
        doc_id = import_ocr_from_s3(
            rep_id=rep_id,
            vol_id=vol_id,
            vol_version=vol_version,
            etext_source=etext_source,
        )
        logger.info("✓ Import completed successfully: %s", doc_id)
    except Exception:
        logger.exception("✗ Import failed for %s/%s", rep_id, vol_id)


@router.post("/ocr-volume")
async def import_ocr_volume(body: ImportOCRRequest, background_tasks: BackgroundTasks) -> dict[str, Any]:
    """Queue an OCR volume import — downloads parquet from S3 and indexes it."""
    logger.info(
        "Queuing import request: %s/%s (version=%s, etext_source=%s)",
        body.rep_id,
        body.vol_id,
        body.vol_version,
        body.etext_source,
    )
    background_tasks.add_task(
        _import_ocr_volume_task,
        rep_id=body.rep_id,
        vol_id=body.vol_id,
        vol_version=body.vol_version,
        etext_source=body.etext_source,
    )
    return {
        "status": "accepted",
        "message": f"Import queued for {body.rep_id}/{body.vol_id}",
    }


def _sync_catalog_task(*, force: bool = False) -> None:
    """Background task that syncs works and persons from BDRC git repos."""
    logger.info("Starting catalog sync (force=%s)", force)
    try:
        entity_scores = load_entity_scores()

        for record_type in ("work", "person"):
            logger.info("Syncing %s records...", record_type)
            counts = sync_repo(record_type, entity_scores, force=force)
            logger.info("Sync %s result: %s", record_type, counts)

        logger.info("Catalog sync completed successfully")
    except Exception:
        logger.exception("Catalog sync failed")


@router.post("/sync-catalog")
async def sync_catalog(
    background_tasks: BackgroundTasks,
    *,
    force: Annotated[bool, Query(description="Force full reimport")] = False,
) -> dict[str, Any]:
    """Import persons and works from the BDRC catalog (incremental)."""
    background_tasks.add_task(_sync_catalog_task, force=force)
    return {
        "status": "accepted",
        "message": "Catalog sync queued",
        "force": force,
    }
