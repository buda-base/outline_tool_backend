"""
Batch deduplication script for detecting duplicate texts across volumes.

Scrolls all volume_etext documents from OpenSearch, extracts text samples
for each segment, queries for similar content in other volumes, and
classifies candidate pairs by wa_id comparison.

Usage:
    python -m scripts.dedup_batch [--output dedup_results.csv] [--dry-run] [--limit N]

Requires the OpenSearch environment variables from .env.
"""

import argparse
import contextlib
import csv
import logging
import time
from pathlib import Path
from typing import Any

from api.config import index_name, opensearch_client
from api.models import DocumentType
from api.services.matching import (
    _extract_text_from_chunks,
    build_matching_query,
    extract_samples,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SCROLL_SIZE = 50
SCROLL_TIMEOUT = "5m"
DEFAULT_OUTPUT = "dedup_results.csv"
MATCH_SCORE_THRESHOLD = 5.0


def scroll_all_volumes() -> list[dict[str, Any]]:
    """Scroll all volume_etext documents from OpenSearch.

    Returns a list of volume docs with id, wa_id, segments, and chunks.
    """
    query_body: dict[str, Any] = {"query": {"term": {"type": DocumentType.VOLUME_ETEXT.value}}}

    volumes: list[dict[str, Any]] = []

    response = opensearch_client.search(
        index=index_name,
        body=query_body,
        size=SCROLL_SIZE,
        scroll=SCROLL_TIMEOUT,
        _source_includes=["wa_id", "mw_id", "segments", "chunks"],
    )

    scroll_id = response.get("_scroll_id")
    hits = response["hits"]["hits"]

    while hits:
        for hit in hits:
            doc = {
                "id": hit["_id"],
                **hit["_source"],
            }
            volumes.append(doc)

        response = opensearch_client.scroll(scroll_id=scroll_id, scroll=SCROLL_TIMEOUT)
        scroll_id = response.get("_scroll_id")
        hits = response["hits"]["hits"]

    if scroll_id:
        with contextlib.suppress(Exception):
            opensearch_client.clear_scroll(scroll_id=scroll_id)

    return volumes


def _extract_segment_info(volume: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract segment info from a volume, including text from chunks.

    Returns a list of dicts with keys:
        volume_id, segment_idx, wa_id, mw_id, title_bo, cstart, cend, text
    """
    segments = volume.get("segments", [])
    chunks = volume.get("chunks", [])
    volume_id = volume["id"]
    volume_wa_id = volume.get("wa_id")

    result: list[dict[str, Any]] = []

    if not segments:
        # Volume has no segments — treat the whole volume as one segment
        if not chunks:
            return []
        full_text = "".join(c.get("text_bo", "") for c in chunks)
        if not full_text.strip():
            return []
        result.append(
            {
                "volume_id": volume_id,
                "segment_idx": 0,
                "wa_id": volume_wa_id,
                "mw_id": volume.get("mw_id"),
                "title_bo": None,
                "cstart": 0,
                "cend": len(full_text),
                "text": full_text,
            }
        )
        return result

    for idx, seg in enumerate(segments):
        cstart = seg.get("cstart", 0)
        cend = seg.get("cend", 0)
        if cend <= cstart:
            continue

        text = _extract_text_from_chunks(chunks, cstart, cend)
        if not text.strip():
            continue

        seg_wa_id = seg.get("wa_id") or volume_wa_id
        title = seg.get("title_bo")
        if isinstance(title, list):
            title = title[0] if title else None

        result.append(
            {
                "volume_id": volume_id,
                "segment_idx": idx,
                "wa_id": seg_wa_id,
                "mw_id": seg.get("mw_id"),
                "title_bo": title,
                "cstart": cstart,
                "cend": cend,
                "text": text,
            }
        )

    return result


def _classify_match(
    source_wa: str | None,
    match_wa: str | None,
) -> str:
    """Classify a match based on wa_id comparison.

    Returns one of:
        - 'propose_merge': different WA, similar content
        - 'confirm_duplicate': same WA, similar content
        - 'warn_mismatch': same WA, but flagged for review
    """
    if source_wa and match_wa:
        if source_wa == match_wa:
            return "confirm_duplicate"
        return "propose_merge"

    # One or both WAs unknown
    return "propose_merge"


def find_duplicates_for_segment(
    segment_info: dict[str, Any],
    *,
    score_threshold: float = MATCH_SCORE_THRESHOLD,
) -> list[dict[str, Any]]:
    """Find duplicate candidates for a single segment.

    Returns a list of candidate dicts ready for CSV output.
    """
    text = segment_info["text"]
    source_volume_id = segment_info["volume_id"]

    samples = extract_samples(text)
    if not samples:
        return []

    query_body = build_matching_query(samples, exclude_volume_id=source_volume_id)

    try:
        response = opensearch_client.search(
            index=index_name,
            body=query_body,
            size=10,
            _source_includes=["wa_id", "mw_id", "segments.wa_id", "segments.mw_id", "segments.title_bo"],
            _source_excludes=["chunks", "pages"],
        )
    except Exception:
        logger.exception(
            "Failed to query for segment %s/%d",
            source_volume_id,
            segment_info["segment_idx"],
        )
        return []

    candidates: list[dict[str, Any]] = []
    for hit in response["hits"]["hits"]:
        match_score = hit.get("_score", 0.0)
        if match_score < score_threshold:
            continue

        match_source = hit["_source"]
        match_volume_id = hit["_id"]

        # Collect all wa_ids from the matched volume's segments
        match_wa_ids: set[str] = set()
        match_wa_id_top = match_source.get("wa_id")
        if match_wa_id_top:
            match_wa_ids.add(match_wa_id_top)
        for seg in match_source.get("segments", []):
            seg_wa = seg.get("wa_id")
            if seg_wa:
                match_wa_ids.add(seg_wa)

        source_wa = segment_info["wa_id"]

        if not match_wa_ids:
            match_wa_ids_resolved: list[str | None] = [None]
        else:
            match_wa_ids_resolved = list(match_wa_ids)

        for match_wa in match_wa_ids_resolved:
            classification = _classify_match(source_wa, match_wa)
            candidates.append(
                {
                    "source_volume_id": source_volume_id,
                    "source_segment_idx": segment_info["segment_idx"],
                    "source_wa_id": source_wa or "",
                    "source_mw_id": segment_info["mw_id"] or "",
                    "source_title_bo": segment_info["title_bo"] or "",
                    "match_volume_id": match_volume_id,
                    "match_wa_id": match_wa or "",
                    "match_score": round(match_score, 3),
                    "classification": classification,
                }
            )

    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch deduplication of volume texts")
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output CSV file path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only count volumes and segments, don't run queries",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit the number of segments to process (0 = no limit)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=MATCH_SCORE_THRESHOLD,
        help=f"Minimum match score threshold (default: {MATCH_SCORE_THRESHOLD})",
    )
    args = parser.parse_args()

    score_threshold = args.threshold

    logger.info("Starting batch deduplication")
    logger.info("Scrolling all volume_etext documents...")

    t0 = time.monotonic()
    volumes = scroll_all_volumes()
    scroll_elapsed = time.monotonic() - t0
    logger.info("Loaded %d volumes in %.1fs", len(volumes), scroll_elapsed)

    # Extract all segments
    all_segments: list[dict[str, Any]] = []
    for vol in volumes:
        segs = _extract_segment_info(vol)
        all_segments.extend(segs)

    logger.info("Extracted %d segments from %d volumes", len(all_segments), len(volumes))

    if args.dry_run:
        logger.info("Dry run complete — would process %d segments", len(all_segments))
        return

    if args.limit > 0:
        all_segments = all_segments[: args.limit]
        logger.info("Limited to %d segments", len(all_segments))

    # Process each segment
    output_path = Path(args.output)
    csv_fields = [
        "source_volume_id",
        "source_segment_idx",
        "source_wa_id",
        "source_mw_id",
        "source_title_bo",
        "match_volume_id",
        "match_wa_id",
        "match_score",
        "classification",
    ]

    total_candidates = 0
    processed = 0
    skipped_short = 0

    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_fields)
        writer.writeheader()

        t_start = time.monotonic()

        for i, seg_info in enumerate(all_segments):
            text_len = len(seg_info["text"])
            if text_len < 50:
                skipped_short += 1
                continue

            candidates = find_duplicates_for_segment(
                seg_info,
                score_threshold=score_threshold,
            )

            for candidate in candidates:
                writer.writerow(candidate)
                total_candidates += 1

            processed += 1

            if processed % 100 == 0:
                elapsed = time.monotonic() - t_start
                rate = processed / elapsed if elapsed > 0 else 0
                eta_seconds = (len(all_segments) - i - 1) / rate if rate > 0 else 0
                eta_minutes = eta_seconds / 60
                logger.info(
                    "[%d/%d] Processed %d segments, %d candidates found (%.1f seg/s, ETA: %.0fm)",
                    i + 1,
                    len(all_segments),
                    processed,
                    total_candidates,
                    rate,
                    eta_minutes,
                )

    elapsed_total = time.monotonic() - t_start
    logger.info("=" * 60)
    logger.info(
        "Dedup complete: %d segments processed, %d skipped (short), %d candidates found in %.1fs",
        processed,
        skipped_short,
        total_candidates,
        elapsed_total,
    )
    logger.info("Results written to %s", output_path.resolve())


if __name__ == "__main__":
    main()
