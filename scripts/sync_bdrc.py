from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ruff: noqa: S603, S607
from api.models import ParsedRecord, SyncCounts
from api.services.os_client import get_document, index_document, refresh_index
from scripts.entity_scores import load_entity_scores
from scripts.import_record import process_parsed_records
from scripts.trig_parser import parse_trig_file

logger = logging.getLogger(__name__)

BDRC_GITLAB_BASE = "https://gitlab.com/bdrc-data"

REPO_CONFIG: dict[str, dict[str, str]] = {
    "work": {
        "repo": "works-20220922",
        "watermark_id": "work_import_record",
    },
    "person": {
        "repo": "persons-20220922",
        "watermark_id": "person_import_record",
    },
}

DEFAULT_DATA_DIR = os.getenv("BDRC_DATA_DIR", "./bdrc_data")


def _clone_or_pull(repo_name: str, data_dir: str) -> Path:
    """Clone the repo if not present, otherwise pull latest."""
    repo_url = f"{BDRC_GITLAB_BASE}/{repo_name}.git"
    repo_path = Path(data_dir) / repo_name

    if repo_path.exists():
        logger.info("Pulling latest for %s", repo_name)
        subprocess.run(
            ["git", "pull", "--ff-only"],
            cwd=str(repo_path),
            check=True,
            capture_output=True,
            text=True,
        )
    else:
        logger.info("Cloning %s from %s", repo_name, repo_url)
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--single-branch", repo_url, str(repo_path)],
            check=True,
            capture_output=True,
            text=True,
        )

    return repo_path


def _get_head_revision(repo_path: Path) -> str:
    """Get the current HEAD commit hash."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo_path),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _revision_exists(repo_path: Path, revision: str) -> bool:
    """Check if a revision exists in the repo."""
    result = subprocess.run(
        ["git", "cat-file", "-t", revision],
        cwd=str(repo_path),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _list_all_trig_files(repo_path: Path) -> list[Path]:
    """List all .trig files in the repo."""
    return sorted(repo_path.rglob("*.trig"))


def _list_changed_trig_files(repo_path: Path, since_revision: str) -> list[Path]:
    """List .trig files changed since a given revision."""
    result = subprocess.run(
        ["git", "diff", "--name-only", f"{since_revision}..HEAD", "--", "*.trig"],
        cwd=str(repo_path),
        check=True,
        capture_output=True,
        text=True,
    )
    changed_files: list[Path] = []
    for line in result.stdout.strip().splitlines():
        file_path = repo_path / line.strip()
        if file_path.exists():
            changed_files.append(file_path)
    return changed_files


def _read_watermark(watermark_id: str) -> dict[str, Any] | None:
    """Read the sync watermark from OpenSearch."""
    return get_document(watermark_id)


def _write_watermark(watermark_id: str, revision: str) -> None:
    """Write the sync watermark to OpenSearch."""
    body: dict[str, Any] = {
        "last_updated_at": datetime.now(UTC).isoformat(),
        "last_revision_imported": revision,
    }
    index_document(watermark_id, body)
    logger.info("Updated watermark %s → revision %s", watermark_id, revision)


def sync_repo(
    record_type: str,
    entity_scores: dict[str, float],
    *,
    force: bool = False,
    data_dir: str = DEFAULT_DATA_DIR,
    limit: int | None = None,
    dry_run: bool = False,
) -> SyncCounts:
    """
    Sync a single repo type (work or person).

    Returns processing counts from process_parsed_records.
    """
    config = REPO_CONFIG[record_type]
    repo_name = config["repo"]
    watermark_id = config["watermark_id"]

    repo_path = _clone_or_pull(repo_name, data_dir)
    head_revision = _get_head_revision(repo_path)

    # Determine which files to import
    watermark = _read_watermark(watermark_id)
    last_revision = None
    if watermark is not None:
        last_revision = watermark.get("last_revision_imported")

    do_full_import = force or last_revision is None or not _revision_exists(repo_path, last_revision)

    if last_revision == head_revision and not force:
        logger.info("Already up to date for %s (revision %s)", record_type, head_revision)
        return SyncCounts()

    if do_full_import:
        logger.info("Full import for %s", record_type)
        trig_files = _list_all_trig_files(repo_path)
    else:
        logger.info(
            "Incremental import for %s: %s..%s",
            record_type,
            last_revision[:8],  # type: ignore[index]
            head_revision[:8],
        )
        trig_files = _list_changed_trig_files(repo_path, last_revision)  # type: ignore[arg-type]

    if not trig_files:
        logger.info("No .trig files to process for %s", record_type)
        _write_watermark(watermark_id, head_revision)
        return SyncCounts()

    if limit is not None:
        trig_files = trig_files[:limit]

    logger.info("Processing %d .trig files for %s", len(trig_files), record_type)

    batch_size = 5000
    now = datetime.now(UTC).isoformat()
    total_counts = SyncCounts()
    parse_errors = 0

    for batch_start in range(0, len(trig_files), batch_size):
        batch = trig_files[batch_start : batch_start + batch_size]
        logger.info(
            "Batch %d-%d of %d",
            batch_start,
            batch_start + len(batch),
            len(trig_files),
        )

        parsed_records: list[ParsedRecord] = []
        for trig_file in batch:
            parsed = parse_trig_file(trig_file)
            if parsed is not None:
                parsed_records.append(parsed)
            else:
                parse_errors += 1

        if dry_run:
            for rec in parsed_records:
                logger.info(
                    "[dry-run] %s %s | released=%s | label=%s | authors=%s",
                    rec.type,
                    rec.id,
                    rec.is_released,
                    rec.pref_label_bo,
                    rec.authors,
                )
            total_counts.skipped += len(parsed_records)
            continue

        counts = process_parsed_records(parsed_records, entity_scores, now=now)
        total_counts.upserted += counts.upserted
        total_counts.merged += counts.merged
        total_counts.withdrawn += counts.withdrawn
        total_counts.skipped += counts.skipped

    if parse_errors:
        logger.warning("Failed to parse %d files", parse_errors)

    if not dry_run:
        refresh_index()
        _write_watermark(watermark_id, head_revision)

    return total_counts


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Sync BDRC works/persons into OpenSearch")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force full reimport (ignore watermark)",
    )
    parser.add_argument(
        "--type",
        choices=["work", "person", "all"],
        default="all",
        help="Which record type to sync (default: all)",
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help=f"Directory for git repos (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of .trig files to process (for testing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse files and log results without writing to OpenSearch",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging.getLogger("opensearch").setLevel(logging.WARNING)

    logger.info("Loading entity scores...")
    entity_scores = load_entity_scores()

    types_to_sync = ["work", "person"] if args.type == "all" else [args.type]

    total = SyncCounts()

    for record_type in types_to_sync:
        logger.info("=== Syncing %s ===", record_type)
        counts = sync_repo(
            record_type,
            entity_scores,
            force=args.force,
            data_dir=args.data_dir,
            limit=args.limit,
            dry_run=args.dry_run,
        )
        total.upserted += counts.upserted
        total.merged += counts.merged
        total.withdrawn += counts.withdrawn
        total.skipped += counts.skipped

    logger.info(
        "=== Sync complete: %d upserted, %d merged, %d withdrawn, %d skipped ===",
        total.upserted,
        total.merged,
        total.withdrawn,
        total.skipped,
    )


if __name__ == "__main__":
    main()
