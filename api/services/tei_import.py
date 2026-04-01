"""Import TEI etext volumes from local XML files into OpenSearch."""

import logging
import re
from datetime import UTC, datetime
from pathlib import Path

from lxml import etree

from api.models import Chunk, DocumentType, PageEntry, VolumeMatchingStatus, VolumeStatus
from api.services.os_client import get_document, index_document
from api.services.volumes import _volume_doc_id
from scripts.tei_to_standoff import convert_tei_root_to_standoff

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1000
ETEXT_SOURCE = "tei"

_TIB_CHUNK_PATTERN = re.compile(r"([སའངགདནབམརལཏ]ོ[་༌]?[།-༔][^ཀ-ཬ]*|(།།|[༎-༒])[^ཀ-ཬ༠-༩]*[།-༔][^ཀ-ཬ༠-༩]*)")


def _build_chunks(text: str, chunk_size: int = CHUNK_SIZE) -> list[Chunk]:
    """Split text into chunks of ~chunk_size chars, breaking at Tibetan sentence endings or newlines."""
    text_len = len(text)
    if text_len <= chunk_size:
        return [Chunk(cstart=0, cend=text_len, text_bo=text)] if text else []

    breaks = [m.end() for m in _TIB_CHUNK_PATTERN.finditer(text)]

    chunks: list[Chunk] = []
    start = 0
    break_index = 0

    while text_len - start > chunk_size:
        target = start + chunk_size
        max_end = min(text_len, start + 2 * chunk_size)

        while break_index < len(breaks) and breaks[break_index] < target:
            break_index += 1

        if break_index > 0 and breaks[break_index - 1] > start:
            end = breaks[break_index - 1]
        elif break_index < len(breaks) and breaks[break_index] <= max_end:
            end = breaks[break_index]
        else:
            newline = text.rfind("\n", start + 1, max_end)
            space = text.rfind(" ", start + 1, max_end)
            best_break = max(newline, space)
            end = best_break + 1 if best_break != -1 else max_end

        chunks.append(Chunk(cstart=start, cend=end, text_bo=text[start:end]))
        start = end

    if start < text_len:
        chunks.append(Chunk(cstart=start, cend=text_len, text_bo=text[start:text_len]))

    return chunks


def parse_tei_volume(xml_path: Path) -> tuple[str, list[PageEntry], dict[str, int], list[dict]] | None:
    """Parse a single TEI XML file and return text, pages, milestones, and etext_spans.

    Args:
        xml_path: Path to the TEI XML file.

    Returns:
        Tuple of (full_text, pages, milestones, etext_spans) or None if parsing fails.
        - full_text: The plain text extracted from the TEI.
        - pages: List of PageEntry objects with character offsets.
        - milestones: Dict mapping milestone ID to character offset.
        - etext_spans: List of dicts with etext_id, cstart, cend for each etext in the volume.
    """
    parser = etree.XMLParser(remove_blank_text=True, remove_comments=True, remove_pis=True)
    try:
        tree = etree.parse(str(xml_path), parser)
    except etree.XMLSyntaxError:
        logger.exception("Failed to parse XML: %s", xml_path)
        return None

    root = tree.getroot()
    text, annotations, _source_path = convert_tei_root_to_standoff(root)

    if text is None or annotations is None:
        logger.warning("No body content in %s", xml_path)
        return None

    # Extract pages from annotations
    pages: list[PageEntry] = [
        PageEntry(
            cstart=page_ann["cstart"],
            cend=page_ann["cend"],
            pnum=page_ann.get("pnum"),
            pname=page_ann.get("pname"),
        )
        for page_ann in annotations.get("pages", [])
    ]

    # Extract milestones
    milestones: dict[str, int] = annotations.get("milestones", {})

    # Build etext_spans from milestone IDs
    # Milestones are section markers like D1, D1-1, D1-2, etc.
    # Each consecutive pair defines an etext span
    etext_spans = _build_etext_spans(milestones, len(text))

    return text, pages, milestones, etext_spans


def _build_etext_spans(milestones: dict[str, int], text_length: int) -> list[dict]:
    """Build etext span entries from milestone annotations.

    Each milestone marks the start of a new section/etext. The span runs
    from this milestone to the next milestone (or end of text for the last one).

    Only considers milestones that look like section IDs (D-prefixed),
    not the synthetic bop:/eop: page boundary milestones.

    Returns:
        List of dicts with keys: milestone_id, cstart, cend
    """
    # Filter to only section milestones (not bop:/eop: page boundaries)
    section_milestones = {
        mid: coord for mid, coord in milestones.items() if not mid.startswith("bop:") and not mid.startswith("eop:")
    }

    if not section_milestones:
        return []

    # Sort by character offset
    sorted_milestones = sorted(section_milestones.items(), key=lambda item: item[1])

    spans = []
    for i, (milestone_id, cstart) in enumerate(sorted_milestones):
        cend = sorted_milestones[i + 1][1] if i < len(sorted_milestones) - 1 else text_length
        spans.append(
            {
                "milestone_id": milestone_id,
                "cstart": cstart,
                "cend": cend,
            }
        )

    return spans


def discover_volumes(ie_dir: Path) -> list[tuple[str, Path]]:
    """Discover all volume directories and their XML files under an IE archive.

    Args:
        ie_dir: Path to the IE directory (e.g., bdrc_data/IE1ER199)

    Returns:
        List of (ve_id, xml_path) tuples, sorted by VE ID.
    """
    archive_dir = ie_dir / "archive"
    if not archive_dir.exists():
        logger.warning("Archive directory not found: %s", archive_dir)
        return []

    volumes = []
    for ve_dir in sorted(archive_dir.iterdir()):
        if not ve_dir.is_dir() or not ve_dir.name.startswith("VE"):
            continue
        xml_files = sorted(ve_dir.glob("*.xml"))
        if not xml_files:
            logger.warning("No XML files in %s", ve_dir)
            continue
        # Each volume directory has exactly one XML file
        volumes.append((ve_dir.name, xml_files[0]))

    return volumes


def import_tei_volume(
    ie_id: str,
    ve_id: str,
    xml_path: Path,
    volume_number: int,
    wa_id: str | None = None,
    mw_id: str | None = None,
    *,
    dry_run: bool = False,
) -> str | None:
    """Import a single TEI volume into OpenSearch.

    Args:
        ie_id: Instance Edition ID (e.g., IE1ER199)
        ve_id: Volume Etext ID (e.g., VE1ER148)
        xml_path: Path to the TEI XML file.
        volume_number: The volume number in the collection.
        wa_id: Work Abstract ID (optional).
        mw_id: Manifestation Work ID (optional).
        dry_run: If True, parse but don't index.

    Returns:
        The document ID of the created/updated volume, or None on failure.
    """
    result = parse_tei_volume(xml_path)
    if result is None:
        return None

    full_text, pages, milestones, etext_spans = result

    if not full_text.strip():
        logger.warning("Empty text extracted from %s", xml_path)
        return None

    # Build search chunks
    chunks = _build_chunks(full_text)

    # Use ie_id as rep_id and ve_id as vol_id for document ID generation
    vol_version = "1"
    doc_id = _volume_doc_id(ie_id, ve_id, vol_version, ETEXT_SOURCE)

    if dry_run:
        logger.info(
            "[dry-run] Would index %s: %d pages, %d milestones, %d etext spans, %d chunks, %d chars",
            doc_id,
            len(pages),
            len(milestones),
            len(etext_spans),
            len(chunks),
            len(full_text),
        )
        return doc_id

    # Check if document already exists to preserve certain fields
    existing_doc = get_document(doc_id)

    now = datetime.now(UTC).isoformat()

    if existing_doc:
        first_imported_at = existing_doc.get("first_imported_at", now)
        existing_segments = existing_doc.get("segments", [])
        existing_status = existing_doc.get("status", VolumeStatus.ACTIVE.value)
        existing_status_matching = existing_doc.get("status_matching", VolumeMatchingStatus.PENDING.value)
        logger.info(
            "Reimporting existing volume %s — preserving %d segments and status=%s",
            doc_id,
            len(existing_segments),
            existing_status,
        )
    else:
        first_imported_at = now
        existing_segments = []
        existing_status = VolumeStatus.ACTIVE.value
        existing_status_matching = VolumeMatchingStatus.PENDING.value
        logger.info("Creating new volume %s", doc_id)

    # Filter milestones to only section milestones for storage.
    # Store as a list of {id, offset} dicts instead of a flat object so that
    # each milestone key does not create a separate OpenSearch field mapping.
    section_milestones = [
        {"id": mid, "offset": coord}
        for mid, coord in milestones.items()
        if not mid.startswith("bop:") and not mid.startswith("eop:")
    ]

    body = {
        "id": doc_id,
        "type": DocumentType.VOLUME_ETEXT.value,
        "rep_id": ie_id,
        "vol_id": ve_id,
        "vol_version": vol_version,
        "etext_source": ETEXT_SOURCE,
        "status": existing_status,
        "status_matching": existing_status_matching,
        "volume_number": volume_number,
        "wa_id": wa_id,
        "mw_id": mw_id,
        "nb_pages": len(pages),
        "pages": [p.model_dump(exclude_none=True) for p in pages],
        "etext_spans": etext_spans,
        "etext_milestones_list": section_milestones,
        "segments": existing_segments,
        "chunks": [c.model_dump() for c in chunks],
        "cstart": 0,
        "cend": len(full_text),
        "first_imported_at": first_imported_at,
        "last_updated_at": now,
    }

    index_document(doc_id, body, refresh=False)

    return doc_id
