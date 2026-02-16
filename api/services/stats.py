from typing import Any

from api.models import CatalogBreakdown, DocumentType, Stats, VolumeStatus
from api.services.os_client import search


def get_stats() -> Stats:
    body: dict[str, Any] = {
        "size": 0,
        "aggs": {
            "by_type": {
                "terms": {"field": "type", "size": 10},
                "aggs": {
                    "by_status": {
                        "terms": {"field": "status", "size": 10},
                    },
                    "total_segments": {
                        "nested": {"path": "segments"},
                        "aggs": {
                            "count": {"value_count": {"field": "segments.cstart"}},
                        },
                    },
                },
            },
        },
    }
    raw = search(body, size=0)
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

        if doc_type == DocumentType.VOLUME_ETEXT.value:
            nb_volumes_imported.no_preexisting_catalog = doc_count

            status_buckets = bucket.get("by_status", {}).get("buckets", [])
            for status_bucket in status_buckets:
                if status_bucket["key"] == VolumeStatus.COMPLETED.value:
                    nb_volumes_finished.no_preexisting_catalog = status_bucket["doc_count"]

            segments_agg = bucket.get("total_segments", {})
            nb_segments_total = segments_agg.get("count", {}).get("value", 0)

        elif doc_type == DocumentType.WORK.value:
            nb_works_total = doc_count

        elif doc_type == DocumentType.PERSON.value:
            nb_persons_total = doc_count

    return Stats(
        nb_volumes_imported=nb_volumes_imported,
        nb_volumes_finished=nb_volumes_finished,
        nb_segments_total=nb_segments_total,
        nb_works_total=nb_works_total,
        nb_persons_total=nb_persons_total,
    )
