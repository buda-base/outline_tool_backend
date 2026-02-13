from fastapi import APIRouter

from api.models import CatalogBreakdown, Stats, VolumeStatus
from api.services.opensearch import get_stats as fetch_raw_stats

router = APIRouter(tags=["stats"])


@router.get("/stats")
async def get_stats() -> Stats:
    """Get aggregated statistics for reporting."""
    raw = fetch_raw_stats()
    aggs = raw.get("aggregations", {})
    by_type = aggs.get("by_type", {}).get("buckets", [])

    nb_volumes_imported = CatalogBreakdown()
    nb_volumes_finished = CatalogBreakdown()
    nb_segments_total = 0
    nb_works_total = 0
    nb_persons_total = 0

    for bucket in by_type:
        doc_type = bucket["key"]
        doc_count = bucket["doc_count"]

        if doc_type == "volume_etext":
            nb_volumes_imported.no_preexisting_catalog = doc_count

            status_buckets = bucket.get("by_status", {}).get("buckets", [])
            for status_bucket in status_buckets:
                if status_bucket["key"] == VolumeStatus.COMPLETED.value:
                    nb_volumes_finished.no_preexisting_catalog = status_bucket["doc_count"]

            segments_agg = bucket.get("total_segments", {})
            segment_count_agg = segments_agg.get("count", {})
            nb_segments_total = segment_count_agg.get("value", 0)

        elif doc_type == "work":
            nb_works_total = doc_count

        elif doc_type == "person":
            nb_persons_total = doc_count

    return Stats(
        nb_volumes_imported=nb_volumes_imported,
        nb_volumes_finished=nb_volumes_finished,
        nb_segments_total=nb_segments_total,
        nb_works_total=nb_works_total,
        nb_persons_total=nb_persons_total,
    )
