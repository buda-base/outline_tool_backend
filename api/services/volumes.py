from datetime import UTC, datetime
from typing import Any

from api.exceptions import NotFoundError
from api.models import (
    AnnotatedSegment,
    DocumentType,
    RecordStatus,
    Segment,
    SegmentType,
    VolumeAnnotationInput,
    VolumeInput,
    VolumeOutput,
    VolumeStatus,
)
from api.services.os_client import extract_hits, get_document, index_document, search, update_document


def _volume_doc_id(rep_id: str, vol_id: str, vol_version: str, etext_source: str) -> str:
    return f"{rep_id}_{vol_id}_{vol_version}_{etext_source}"


def list_volumes(
    status: str | None = None,
    etext_source: str | None = None,
    rep_id: str | None = None,
    offset: int = 0,
    limit: int = 50,
) -> tuple[list[VolumeOutput], int]:
    filters: list[dict[str, Any]] = [
        {"term": {"type": DocumentType.VOLUME_ETEXT.value}},
    ]
    if status is not None:
        filters.append({"term": {"status": status}})
    if etext_source is not None:
        filters.append({"term": {"etext_source": etext_source}})
    if rep_id is not None:
        filters.append({"term": {"rep_id": rep_id}})

    body: dict[str, Any] = {"query": {"bool": {"filter": filters}}}
    response = search(
        body,
        size=limit,
        offset=offset,
        source_excludes=["chunks", "pages", "segments"],
    )
    total: int = response["hits"]["total"]["value"]
    items = [VolumeOutput.model_validate(h) for h in extract_hits(response)]
    return items, total


def get_volume(rep_id: str, vol_id: str) -> VolumeOutput | None:
    """
    Get the best matching volume for the given rep_id and vol_id.

    Selection logic:
    1. Prefer documents with segments over those without
    2. Within each group, prefer most recent (by last_updated_at)
    """
    # Search for all volumes matching rep_id and vol_id
    filters: list[dict[str, Any]] = [
        {"term": {"type": DocumentType.VOLUME_ETEXT.value}},
        {"term": {"rep_id": rep_id}},
        {"term": {"vol_id": vol_id}},
    ]

    body: dict[str, Any] = {
        "query": {"bool": {"filter": filters}},
        "sort": [{"last_updated_at": {"order": "desc"}}],
    }

    response = search(body, size=100)  # Get up to 100 versions
    hits = extract_hits(response)

    if not hits:
        return None

    # Separate documents with and without segments
    with_segments = [h for h in hits if h.get("segments") and len(h.get("segments", [])) > 0]
    without_segments = [h for h in hits if not h.get("segments") or len(h.get("segments", [])) == 0]

    # Choose the best: prefer with segments, then most recent
    chosen = with_segments[0] if with_segments else (without_segments[0] if without_segments else hits[0])

    return VolumeOutput.model_validate({**chosen, "id": chosen["id"], "rep_id": rep_id, "vol_id": vol_id})


def update_volume(rep_id: str, vol_id: str, data: VolumeInput) -> VolumeOutput:
    """
    Partial update of an existing volume (only client-sent fields).
    Requires vol_version and etext_source to identify the specific document to update.
    """
    if not data.vol_version or not data.etext_source:
        raise ValueError("vol_version and etext_source are required to update a volume")

    doc_id = _volume_doc_id(rep_id, vol_id, data.vol_version, data.etext_source)

    existing = get_document(doc_id)
    if existing is None:
        raise NotFoundError("Volume", doc_id)

    partial = data.model_dump(exclude_unset=True)
    partial["last_updated_at"] = datetime.now(UTC).isoformat()
    update_document(doc_id, partial)
    return VolumeOutput.model_validate({**existing, **partial, "id": doc_id})


def save_annotated_volume(volume_id: str, data: VolumeAnnotationInput) -> str:
    """
    Save fully annotated volume from frontend.
    
    Args:
        volume_id: The OpenSearch document ID (e.g., W00CHZ0103341_I1CZ35_822f2e_ocrv1-ws-ldv1)
        data: The annotated volume data from frontend
    
    Returns:
        The document ID
    
    Raises:
        ValueError: If validation fails or document doesn't exist
    """
    # Check if document exists
    existing = get_document(volume_id)
    if existing is None:
        raise ValueError(f"Volume with ID {volume_id} not found")
    
    # Validate that the parent mw_id from document matches segment mw_ids
    parent_mw_id = existing.get("mw_id")
    if parent_mw_id:
        for segment in data.segments:
            if not segment.mw_id.startswith(f"{parent_mw_id}_"):
                raise ValueError(
                    f"Segment mw_id '{segment.mw_id}' must start with parent mw_id '{parent_mw_id}_'"
                )
    
    # Convert AnnotatedSegment to internal Segment format
    segments = []
    for seg in data.segments:
        # Normalize title_bo and author_name_bo to lists
        title_bo_list = seg.title_bo if isinstance(seg.title_bo, list) else [seg.title_bo]
        author_name_bo_list = None
        if seg.author_name_bo:
            author_name_bo_list = (
                seg.author_name_bo if isinstance(seg.author_name_bo, list) else [seg.author_name_bo]
            )
        
        # Create internal segment representation
        internal_seg = {
            "mw_id": seg.mw_id,
            "wa_id": seg.wa_id,
            "cstart": seg.cstart,
            "cend": seg.cend,
            "part_type": seg.part_type.value,
            "title_bo": title_bo_list,
        }
        
        if author_name_bo_list:
            internal_seg["author_name_bo"] = author_name_bo_list
        
        segments.append(internal_seg)
    
    # Update document with new data
    update_data = {
        "record_status": data.record_status.value,
        "base_text": data.base_text,
        "segments": segments,
        "last_updated_at": datetime.now(UTC).isoformat(),
    }
    
    update_document(volume_id, update_data)
    
    return volume_id
