"""
Import TEI etext volumes into OpenSearch.

Phase 1: Parse TEI XML files from {ie_dir}/archive/VE*/,
         extract text + pagination via tei_to_standoff, build volume
         documents with pages, etext_spans, milestones, and chunks,
         and index them into OpenSearch.

Phase 2: Read the CSV outline file, parse segment rows with etext
         coordinates (vol_num#milestone_id), resolve milestone IDs to
         character offsets from the indexed volume documents, and store
         segments on the volume documents.

Usage:
    # Full import (both phases) for Kangyur (uses built-in defaults)
    python -m scripts.import_tei --ie-id IE1ER199

    # Full import for Tengyur
    python -m scripts.import_tei --ie-id IE1ER200

    # Custom IE with explicit parameters
    python -m scripts.import_tei --ie-id IE_CUSTOM --ie-dir /path/to/data \
        --wa-id WA0XYZ --mw-id MW0XYZ --csv-path /path/to/outline.csv

    # Phase 1 only (TEI volume import)
    python -m scripts.import_tei --ie-id IE1ER199 --phase volumes

    # Phase 2 only (segment import from CSV)
    python -m scripts.import_tei --ie-id IE1ER199 --phase segments

    # Dry run
    python -m scripts.import_tei --ie-id IE1ER199 --dry-run
"""

import argparse
import csv
import logging
import os
from pathlib import Path
from typing import Any

from api.models import SegmentType
from api.services.os_client import get_document, refresh_index, update_document
from api.services.tei_import import (
    ETEXT_SOURCE,
    discover_volumes,
    import_tei_volume,
)
from api.services.volumes import _volume_doc_id

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _filter_os_404(record: logging.LogRecord) -> bool:
    """Suppress expected 404 warnings from opensearch client."""
    return "status:404" not in record.getMessage()


logging.getLogger("opensearch").addFilter(_filter_os_404)

DEFAULT_DATA_DIR = os.getenv("BDRC_DATA_DIR", "./bdrc_data")

# Known IE configurations — used as defaults when --wa-id / --mw-id are not given
IE_DEFAULTS: dict[str, dict[str, str | None]] = {
    "IE1ER199": {
        "name": "Kangyur (sDe dge)",
        "wa_id": "WA0BC001",
        "mw_id": "MW22084",
    },
    "IE1ER200": {
        "name": "Tengyur (sDe dge)",
        "wa_id": "WA0BC002",
        "mw_id": "MW22084",
    },
}


# ── Phase 1: Volume import ─────────────────────────────────────────────────


def build_volnum_to_ve_map(ie_dir: Path) -> dict[int, tuple[str, Path]]:
    """Build a mapping from volume number (1-indexed) to (ve_id, xml_path).

    Volume number is the ordinal position of the VE directory when sorted.
    """
    volumes = discover_volumes(ie_dir)
    return {idx: (ve_id, xml_path) for idx, (ve_id, xml_path) in enumerate(volumes, start=1)}


def phase1_import_volumes(
    ie_id: str,
    ie_dir: Path,
    *,
    wa_id: str | None = None,
    mw_id: str | None = None,
    dry_run: bool = False,
    start_from: int = 0,
) -> dict[int, str]:
    """Import all TEI volumes for an IE into OpenSearch.

    Returns:
        Dict mapping volume_number -> doc_id for successfully imported volumes.
    """
    if not ie_dir.exists():
        logger.error("IE directory not found: %s", ie_dir)
        return {}

    volnum_to_ve = build_volnum_to_ve_map(ie_dir)
    total = len(volnum_to_ve)
    logger.info("Found %d volumes for %s in %s", total, ie_id, ie_dir)

    imported: dict[int, str] = {}
    failed = 0

    for vol_num, (ve_id, xml_path) in sorted(volnum_to_ve.items()):
        if vol_num - 1 < start_from:
            continue

        try:
            doc_id = import_tei_volume(
                ie_id=ie_id,
                ve_id=ve_id,
                xml_path=xml_path,
                volume_number=vol_num,
                wa_id=wa_id,
                mw_id=mw_id,
                dry_run=dry_run,
            )
            if doc_id:
                imported[vol_num] = doc_id
                logger.info(
                    "[%d/%d] ✓ Imported volume %d (%s) -> %s",
                    len(imported),
                    total,
                    vol_num,
                    ve_id,
                    doc_id,
                )
            else:
                failed += 1
                logger.warning("[%d/%d] ✗ Failed to import volume %d (%s)", len(imported), total, vol_num, ve_id)
        except Exception:
            logger.exception("Failed to import volume %d (%s)", vol_num, ve_id)
            failed += 1

    if imported and not dry_run:
        logger.info("Refreshing index...")
        refresh_index()

    logger.info("=" * 60)
    logger.info(
        "Phase 1 complete: %d volumes imported, %d failed out of %d total",
        len(imported),
        failed,
        total,
    )

    return imported


# ── CSV parsing ────────────────────────────────────────────────────────────


def _parse_etext_coord(coord: str) -> tuple[int, str] | None:
    """Parse an etext coordinate like '1#D1' into (vol_num, milestone_id).

    Returns None if the coordinate is empty or malformed.
    """
    if not coord or "#" not in coord:
        return None
    parts = coord.split("#", maxsplit=1)
    try:
        vol_num = int(parts[0])
    except ValueError:
        return None
    milestone_id = parts[1]
    return vol_num, milestone_id


def parse_outline_csv(csv_path: Path) -> list[dict[str, Any]]:
    """Parse the outline CSV into a list of structured entries.

    Returns a list of dicts, one per text (T) row, with:
        - label: text label (e.g., 'D1', 'D1-1')
        - volume_number: volume number from the parent V row
        - etext_start: parsed (vol_num, milestone_id) or None
        - etext_end: parsed (vol_num, milestone_id) or None
    """
    entries: list[dict[str, Any]] = []
    current_volume_number: int | None = None

    with csv_path.open(encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)

        # Find column indices by header name
        col_map: dict[str, int] = {}
        for i, name in enumerate(header):
            if name == "part type":
                col_map["part_type"] = i
            elif name == "label":
                col_map["label"] = i
            elif name == "etext start":
                col_map["etext_start"] = i
            elif name == "etext end":
                col_map["etext_end"] = i
            elif name == "img grp start":
                col_map["img_grp_start"] = i

        for row in reader:
            if len(row) <= max(col_map.values()):
                continue

            part_type = row[col_map["part_type"]].strip()
            label = row[col_map["label"]].strip()

            if part_type == "V":
                # Volume row — extract volume number from img grp start
                img_grp_start = row[col_map["img_grp_start"]].strip()
                try:
                    current_volume_number = int(img_grp_start)
                except ValueError:
                    logger.warning("Invalid volume number in CSV: %s", img_grp_start)
                    current_volume_number = None

            elif part_type == "T" and current_volume_number is not None:
                # Text row — extract etext coordinates
                etext_start_raw = row[col_map["etext_start"]].strip()
                etext_end_raw = row[col_map["etext_end"]].strip()

                entries.append(
                    {
                        "label": label,
                        "volume_number": current_volume_number,
                        "etext_start": _parse_etext_coord(etext_start_raw),
                        "etext_end": _parse_etext_coord(etext_end_raw),
                    }
                )

    return entries


# ── Phase 2: Segment import ────────────────────────────────────────────────


def _milestone_to_char(
    milestone_id: str,
    milestones: dict[str, int],
    fallback: int,
    label: str,
) -> int:
    """Map a milestone ID to a character offset, logging if the milestone is missing."""
    if milestone_id in milestones:
        return milestones[milestone_id]
    logger.warning("%s milestone '%s' not found in volume milestones", label, milestone_id)
    return fallback


def phase2_import_segments(
    ie_id: str,
    ie_dir: Path,
    csv_path: Path,
    *,
    dry_run: bool = False,
) -> None:
    """Parse the outline CSV and store segments on volume documents using milestone-based coordinates."""
    if not csv_path.exists():
        logger.error("CSV file not found: %s", csv_path)
        return

    # Parse CSV into structured entries
    entries = parse_outline_csv(csv_path)
    logger.info("Parsed %d text entries from %s", len(entries), csv_path)

    if not entries:
        return

    # Build volume number -> VE ID mapping
    volnum_to_ve = build_volnum_to_ve_map(ie_dir)

    # Group entries by volume number
    entries_by_volume: dict[int, list[dict[str, Any]]] = {}
    for entry in entries:
        vol_num = entry["volume_number"]
        entries_by_volume.setdefault(vol_num, []).append(entry)

    segments_imported = 0
    no_volume_doc = 0
    skipped_has_segments = 0
    failed = 0

    for vol_num in sorted(entries_by_volume):
        vol_entries = entries_by_volume[vol_num]

        ve_info = volnum_to_ve.get(vol_num)
        if ve_info is None:
            logger.warning("No VE directory for volume %d — skipping %d entries", vol_num, len(vol_entries))
            no_volume_doc += len(vol_entries)
            continue

        ve_id = ve_info[0]
        doc_id = _volume_doc_id(ie_id, ve_id, "1", ETEXT_SOURCE)

        if dry_run:
            logger.info(
                "[dry-run] Would import %d segments for volume %d (%s -> %s)",
                len(vol_entries),
                vol_num,
                ve_id,
                doc_id,
            )
            segments_imported += 1
            continue

        try:
            existing = get_document(doc_id)
            if existing is None:
                logger.warning("Volume document %s not found in OpenSearch — skipping", doc_id)
                no_volume_doc += 1
                continue

            # Skip if already has segments
            existing_segments = existing.get("segments", [])
            if existing_segments:
                skipped_has_segments += 1
                continue

            # Get milestones from the volume document.
            # Stored as [{id, offset}, ...]; convert to {id: offset} dict.
            raw_milestones = existing.get("etext_milestones_list", [])
            if isinstance(raw_milestones, list):
                milestones = {m["id"]: m["offset"] for m in raw_milestones}
            else:
                milestones = raw_milestones
            max_cend = existing.get("cend", 0)

            segments = _build_segments_from_csv(vol_entries, milestones, max_cend)

            if not segments:
                logger.warning("No valid segments built for volume %d (%s)", vol_num, doc_id)
                continue

            update_document(doc_id, {"segments": segments}, refresh=False)
            segments_imported += 1
            logger.info(
                "✓ Imported %d segments for volume %d (%s)",
                len(segments),
                vol_num,
                doc_id,
            )
        except Exception:
            logger.exception("Failed to process segments for volume %d (%s)", vol_num, doc_id)
            failed += 1

    if segments_imported and not dry_run:
        logger.info("Refreshing index...")
        refresh_index()

    logger.info("=" * 60)
    logger.info(
        "Phase 2 complete: %d volumes got segments, %d no volume doc, %d already had segments, %d failed",
        segments_imported,
        no_volume_doc,
        skipped_has_segments,
        failed,
    )


def _build_segments_from_csv(
    vol_entries: list[dict[str, Any]],
    milestones: dict[str, int],
    max_cend: int,
) -> list[dict[str, Any]]:
    """Build segment dicts from CSV entries using milestone-based character offsets.

    Args:
        vol_entries: List of parsed CSV text entries for one volume.
        milestones: Dict mapping milestone ID to character offset (from volume doc).
        max_cend: End of text character offset.

    Returns:
        List of segment dicts ready to store on the volume document.
    """
    segments: list[dict[str, Any]] = []

    for entry in vol_entries:
        etext_start = entry["etext_start"]
        etext_end = entry["etext_end"]

        # Resolve start coordinate
        if etext_start is not None:
            _vol_num, start_milestone = etext_start
            cstart = _milestone_to_char(start_milestone, milestones, 0, "Start")
        else:
            cstart = 0

        # Resolve end coordinate
        if etext_end is not None:
            _vol_num, end_milestone = etext_end
            cend = _milestone_to_char(end_milestone, milestones, max_cend, "End")
        else:
            cend = max_cend

        segment_dict: dict[str, Any] = {
            "cstart": cstart,
            "cend": cend,
            "segment_type": SegmentType.TEXT.value,
        }

        if entry["label"]:
            segment_dict["title_bo"] = [entry["label"]]

        segments.append(segment_dict)

    return segments


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Import TEI etext volumes and outline segments")
    parser.add_argument(
        "--ie-id",
        required=True,
        help="Instance Edition ID (e.g., IE1ER199 for Kangyur, IE1ER200 for Tengyur)",
    )
    parser.add_argument(
        "--phase",
        choices=["volumes", "segments", "both"],
        default="both",
        help="Which phase to run (default: both)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Parse but don't index")
    parser.add_argument(
        "--start-from",
        type=int,
        default=1,
        help="1-based volume number to resume from (Phase 1 only)",
    )
    parser.add_argument(
        "--ie-dir",
        default=None,
        help="Path to the IE directory containing archive/ and CSV. Defaults to {data-dir}/{ie-id}",
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help=f"Base directory for IE data (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--wa-id",
        default=None,
        help="Work Abstract ID (e.g., WA0BC001). Overrides built-in default for known IEs.",
    )
    parser.add_argument(
        "--mw-id",
        default=None,
        help="Manifestation Work ID (e.g., MW22084). Overrides built-in default for known IEs.",
    )
    parser.add_argument(
        "--csv-path",
        default=None,
        help="Path to the outline CSV file. Defaults to {ie-dir}/{ie-id}.csv",
    )
    args = parser.parse_args()

    ie_id = args.ie_id
    defaults = IE_DEFAULTS.get(ie_id, {})

    # Resolve ie_dir: explicit > data_dir/ie_id
    ie_dir = Path(args.ie_dir) if args.ie_dir else Path(args.data_dir) / ie_id

    # Resolve wa_id / mw_id: CLI > built-in defaults
    wa_id = args.wa_id or defaults.get("wa_id")
    mw_id = args.mw_id or defaults.get("mw_id")

    # Resolve CSV path: CLI > ie_dir/{ie_id}.csv
    csv_path = Path(args.csv_path) if args.csv_path else ie_dir / f"{ie_id}.csv"

    name = defaults.get("name", ie_id)
    logger.info("Importing %s (%s) from %s", ie_id, name, ie_dir)
    if wa_id or mw_id:
        logger.info("  wa_id=%s, mw_id=%s", wa_id, mw_id)

    start_idx = max(args.start_from - 1, 0)

    if args.phase in ("volumes", "both"):
        logger.info("=== Phase 1: Import TEI volumes ===")
        phase1_import_volumes(
            ie_id,
            ie_dir,
            wa_id=wa_id,
            mw_id=mw_id,
            dry_run=args.dry_run,
            start_from=start_idx,
        )

    if args.phase in ("segments", "both"):
        logger.info("=== Phase 2: Import segments from CSV ===")
        phase2_import_segments(
            ie_id,
            ie_dir,
            csv_path,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
