"""
Batch import script for volumes from S3 parquet files into OpenSearch.

Reads a CSV file (w_id, i_id, i_version, etext_source) and imports
each volume from the S3 parquet files into OpenSearch.

Usage:
    python -m scripts.import_batch path/to/batch.csv [--dry-run] [--start-from INDEX]

Requires AWS credentials configured (for S3 access) and the OpenSearch
environment variables from .env.
"""

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

from api.services.ocr_import import import_ocr_from_s3
from api.services.os_client import get_document
from api.services.volumes import _volume_doc_id

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_csv(path: str) -> list[dict[str, str]]:
    """Load the batch CSV (no header row) into a list of dicts."""
    rows: list[dict[str, str]] = []
    with Path(path).open(newline="") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row or not row[0].strip():
                continue
            rows.append(
                {
                    "w_id": row[0].strip(),
                    "i_id": row[1].strip(),
                    "i_version": row[2].strip(),
                    "etext_source": row[3].strip(),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Import volumes from S3 parquet files")
    parser.add_argument("csv", help="Path to the batch CSV file (w_id,i_id,i_version,etext_source)")
    parser.add_argument("--dry-run", action="store_true", help="Only list what would be imported")
    parser.add_argument("--force", action="store_true", help="Reimport volumes even if already indexed")
    parser.add_argument(
        "--start-from",
        type=int,
        default=0,
        help="0-based row index to resume from (skip earlier rows)",
    )
    args = parser.parse_args()

    rows = load_csv(args.csv)
    logger.info("Loaded %d rows from %s", len(rows), args.csv)

    if args.dry_run:
        for i, row in enumerate(rows):
            print(f"[{i:4d}] {row['w_id']}  {row['i_id']}  {row['i_version']}  {row['etext_source']}")  # noqa: T201
        logger.info("Dry run complete — %d volumes would be imported", len(rows))
        return

    succeeded = 0
    skipped = 0
    failed = 0
    failed_rows: list[tuple[int, dict[str, str], str]] = []

    for i, row in enumerate(rows):
        if i < args.start_from:
            continue

        w_id = row["w_id"]
        i_id = row["i_id"]
        i_version = row["i_version"]
        etext_source = row["etext_source"]

        if not args.force:
            doc_id = _volume_doc_id(w_id, i_id, i_version, etext_source)
            if get_document(doc_id) is not None:
                logger.info("[%d/%d] Skipping %s (already indexed)", i + 1, len(rows), doc_id)
                skipped += 1
                continue

        logger.info(
            "[%d/%d] Importing %s / %s / %s / %s",
            i + 1,
            len(rows),
            w_id,
            i_id,
            i_version,
            etext_source,
        )

        try:
            t0 = time.monotonic()
            doc_id = import_ocr_from_s3(w_id, i_id, i_version, etext_source)
            elapsed = time.monotonic() - t0
            logger.info("  ✓ Indexed as %s  (%.1fs)", doc_id, elapsed)
            succeeded += 1
        except Exception:
            logger.exception("  ✗ Failed to import row %d (%s / %s)", i, w_id, i_id)
            failed += 1
            failed_rows.append((i, row, str(sys.exc_info()[1])))

    logger.info("=" * 60)
    logger.info(
        "Import complete: %d succeeded, %d skipped, %d failed out of %d",
        succeeded,
        skipped,
        failed,
        len(rows),
    )

    if failed_rows:
        logger.warning("Failed rows:")
        for idx, row, err in failed_rows:
            logger.warning(
                "  [%d] %s / %s / %s / %s — %s",
                idx,
                row["w_id"],
                row["i_id"],
                row["i_version"],
                row["etext_source"],
                err,
            )


if __name__ == "__main__":
    main()
