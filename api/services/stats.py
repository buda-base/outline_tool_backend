from typing import Any

from api.models import CatalogBreakdown, DocumentType, Stats, VolumeStatus
from api.services.os_client import search


def get_volume_batch_status_report(*, max_batches: int = 5000) -> dict[str, dict[str, int]]:
    """
    Count volume_etext documents per ``batch_id`` and per annotation ``status``.

    Excludes documents with missing or empty ``batch_id``.
    """
    body: dict[str, Any] = {
        "size": 0,
        "query": {
            "bool": {
                "filter": [
                    {"term": {"type": DocumentType.VOLUME_ETEXT.value}},
                    {"exists": {"field": "batch_id"}},
                    {"bool": {"must_not": [{"term": {"batch_id": ""}}]}},
                ],
            },
        },
        "aggs": {
            "batches": {
                "terms": {
                    "field": "batch_id",
                    "size": max(1, min(max_batches, 20000)),
                    "order": {"_key": "asc"},
                },
                "aggs": {
                    "by_status": {
                        "terms": {
                            "field": "status",
                            "size": 32,
                        }
                    }
                },
            }
        },
    }
    raw = search(body, size=0)
    batch_buckets = raw.get("aggregations", {}).get("batches", {}).get("buckets", [])

    result: dict[str, dict[str, int]] = {}
    for batch in batch_buckets:
        batch_key = str(batch.get("key", ""))
        status_map: dict[str, int] = {}
        for sb in batch.get("by_status", {}).get("buckets", []):
            sk = sb.get("key")
            if sk is not None:
                status_map[str(sk)] = int(sb.get("doc_count", 0))
        if status_map:
            result[batch_key] = status_map
        else:
            result[batch_key] = {}

    return result


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
                if status_bucket["key"] == VolumeStatus.REVIEWED.value:
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
