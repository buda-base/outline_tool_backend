import logging
from collections import defaultdict
from typing import Any

from api.models import DocumentType, MatchCandidate, MatchedVolume, PersonOutput
from api.services.os_client import get_document, mget_documents, search

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_SIZE = 200
DEFAULT_MAX_CANDIDATES = 20
TITLE_MATCH_BOOST = 0.3


def extract_samples(text: str, sample_size: int = DEFAULT_SAMPLE_SIZE) -> dict[str, str]:
    text = text.strip()
    text_len = len(text)

    if text_len == 0:
        return {}

    if text_len <= sample_size:
        return {"beginning": text}

    samples: dict[str, str] = {}

    samples["beginning"] = text[:sample_size]

    if text_len > sample_size * 2:
        mid_start = (text_len - sample_size) // 2
        samples["middle"] = text[mid_start : mid_start + sample_size]

    samples["end"] = text[-sample_size:]

    return samples


def _build_chunk_phrase_query(sample_text: str, *, boost: float = 1.0) -> dict[str, Any]:
    return {
        "nested": {
            "path": "chunks",
            "score_mode": "max",
            "query": {
                "match_phrase": {
                    "chunks.text_bo": {
                        "query": sample_text,
                        "boost": boost,
                    }
                }
            },
        }
    }


def _build_title_match_query(sample_text: str, *, boost: float = TITLE_MATCH_BOOST) -> list[dict[str, Any]]:
    return [
        {
            "multi_match": {
                "type": "phrase",
                "query": sample_text,
                "fields": ["pref_label_bo", "alt_label_bo"],
                "boost": boost,
            }
        },
        {
            "nested": {
                "path": "segments",
                "score_mode": "max",
                "query": {
                    "match_phrase": {
                        "segments.title_bo": {
                            "query": sample_text,
                            "boost": boost,
                        }
                    }
                },
            }
        },
    ]


def build_matching_query(
    samples: dict[str, str],
    *,
    exclude_volume_id: str | None = None,
) -> dict[str, Any]:
    should_clauses: list[dict[str, Any]] = []

    for sample_name, sample_text in samples.items():
        boost = 1.2 if sample_name == "beginning" else 1.0
        should_clauses.append(_build_chunk_phrase_query(sample_text, boost=boost))

    if "beginning" in samples:
        should_clauses.extend(_build_title_match_query(samples["beginning"]))

    filters: list[dict[str, Any]] = [
        {"term": {"type": DocumentType.VOLUME_ETEXT.value}},
    ]
    if exclude_volume_id:
        filters.append({"bool": {"must_not": {"term": {"_id": exclude_volume_id}}}})

    query: dict[str, Any] = {
        "query": {
            "bool": {
                "should": should_clauses,
                "minimum_should_match": 1,
                "filter": filters,
            }
        }
    }

    return query


def _extract_text_from_chunks(chunks: list[dict[str, Any]], cstart: int, cend: int) -> str:
    parts: list[str] = []
    for chunk in chunks:
        chunk_start = chunk.get("cstart", 0)
        chunk_end = chunk.get("cend", 0)
        chunk_text = chunk.get("text_bo", "")

        if chunk_end <= cstart or chunk_start >= cend:
            continue

        slice_start = max(0, cstart - chunk_start)
        slice_end = min(len(chunk_text), cend - chunk_start)
        parts.append(chunk_text[slice_start:slice_end])

    return "".join(parts)


def _group_hits_by_wa(hits: list[dict[str, Any]]) -> list[MatchCandidate]:
    wa_groups: dict[str, dict[str, Any]] = defaultdict(lambda: {"score": 0.0, "pref_label_bo": None, "volumes": []})

    for hit in hits:
        volume_id = hit.get("id", "")
        score = hit.get("_score", 0.0)

        segments = hit.get("segments", [])
        wa_ids_in_volume: set[str] = set()
        for seg in segments:
            wa_id = seg.get("wa_id")
            if wa_id:
                wa_ids_in_volume.add(wa_id)

        volume_wa_id = hit.get("wa_id")
        if volume_wa_id:
            wa_ids_in_volume.add(volume_wa_id)

        if not wa_ids_in_volume:
            wa_ids_in_volume = {f"_unknown_{volume_id}"}

        for wa_id in wa_ids_in_volume:
            group = wa_groups[wa_id]
            group["score"] = max(group["score"], score)
            group["volumes"].append(MatchedVolume(volume_id=volume_id, score=score))

            if group["pref_label_bo"] is None:
                for seg in segments:
                    if seg.get("wa_id") == wa_id:
                        title = seg.get("title_bo")
                        if isinstance(title, list) and title:
                            group["pref_label_bo"] = title[0]
                        elif isinstance(title, str):
                            group["pref_label_bo"] = title
                        break

    candidates: list[MatchCandidate] = []
    for wa_id, group in wa_groups.items():
        display_wa_id = wa_id if not wa_id.startswith("_unknown_") else None
        candidates.append(
            MatchCandidate(
                wa_id=display_wa_id,
                pref_label_bo=group["pref_label_bo"],
                score=group["score"],
                matched_volumes=group["volumes"],
            )
        )

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def _resolve_candidates(candidates: list[MatchCandidate]) -> list[MatchCandidate]:
    """Fetch actual work records to fill pref_label_bo and resolve author_records."""
    wa_ids = [c.wa_id for c in candidates if c.wa_id is not None]
    if not wa_ids:
        return candidates

    work_map = mget_documents(wa_ids)

    all_author_ids = list({aid for doc in work_map.values() for aid in doc.get("authors", [])})
    person_map = mget_documents(all_author_ids)

    for candidate in candidates:
        if candidate.wa_id is None or candidate.wa_id not in work_map:
            continue
        work_doc = work_map[candidate.wa_id]

        if candidate.pref_label_bo is None:
            candidate.pref_label_bo = work_doc.get("pref_label_bo")

        author_ids = work_doc.get("authors", [])
        candidate.author_records = [
            PersonOutput.model_validate({**person_map[aid], "id": aid}) for aid in author_ids if aid in person_map
        ]

    return candidates


def find_matching_works(
    text: str,
    *,
    exclude_volume_id: str | None = None,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> list[MatchCandidate]:
    samples = extract_samples(text)
    if not samples:
        return []

    query_body = build_matching_query(samples, exclude_volume_id=exclude_volume_id)

    logger.info(
        "Running matching query with %d samples (sizes: %s), exclude=%s",
        len(samples),
        {k: len(v) for k, v in samples.items()},
        exclude_volume_id,
    )

    response = search(
        query_body,
        size=max_candidates * 3,
        source_excludes=["chunks", "pages"],
    )

    hits = [{**hit["_source"], "id": hit["_id"], "_score": hit.get("_score", 0.0)} for hit in response["hits"]["hits"]]

    logger.info("Matching query returned %d hits", len(hits))

    return _resolve_candidates(_group_hits_by_wa(hits)[:max_candidates])


def find_matching_works_by_volume_ref(
    volume_id: str,
    cstart: int,
    cend: int,
    *,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> list[MatchCandidate]:
    source = get_document(volume_id)
    if source is None:
        raise ValueError(f"Volume '{volume_id}' not found")

    chunks = source.get("chunks", [])
    if not chunks:
        raise ValueError(f"Volume '{volume_id}' has no chunks")

    text = _extract_text_from_chunks(chunks, cstart, cend)
    if not text.strip():
        raise ValueError(f"No text found in volume '{volume_id}' for range [{cstart}, {cend})")

    return find_matching_works(
        text,
        exclude_volume_id=volume_id,
        max_candidates=max_candidates,
    )
