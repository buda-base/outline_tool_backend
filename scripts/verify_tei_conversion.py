"""
Verify TEI-to-standoff conversion against original source .txt files.

For each volume, this script:
  1. Runs convert_tei_root_to_standoff on the archive XML
  2. Parses the source .txt to extract expected milestones and plain text
  3. Compares:
     - Tibetan text content (stripping markers from source)
     - Milestone IDs (curly-brace markers vs annotation milestones)
     - Page labels (bracket markers vs page annotations)
  4. Prints a per-volume pass/fail summary

Usage:
    python -m scripts.verify_tei_conversion --ie-id IE1ER199
    python -m scripts.verify_tei_conversion --ie-id IE1ER200
    python -m scripts.verify_tei_conversion --ie-id IE1ER199 --limit 5
"""

import argparse
import logging
import os
import re
import unicodedata
from pathlib import Path

from lxml import etree

from api.services.tei_import import discover_volumes
from scripts.tei_to_standoff import convert_tei_root_to_standoff

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = os.getenv("BDRC_DATA_DIR", "./bdrc_data")

# Regex patterns for the source .txt format
_PAGE_MARKER_RE = re.compile(r"\[(\d+[ab])\]")
_LINE_MARKER_RE = re.compile(r"\[(\d+[ab]\.\d+)\]")
# Section milestones like {D1}, {D1-1} — no comma inside braces
_MILESTONE_RE = re.compile(r"\{([^},]+)\}")
# Variant readings: {orig,corr} or (orig,corr) — orig appears inline before the marker
_VARIANT_CURLY_RE = re.compile(r"\{([^},]+),([^}]+)\}")
_VARIANT_PAREN_RE = re.compile(r"\(([^),]+),([^)]+)\)")
# Bracket markers (pages/lines)
_BRACKET_MARKER_RE = re.compile(r"\[[^\]]+\]")


def parse_source_txt(txt_path: Path) -> tuple[str, list[str], list[str]]:
    """Parse a source .txt file, extracting text, milestone IDs, and page labels.

    Returns:
        (plain_text, milestone_ids, page_labels)
        - plain_text: The text with all bracket and curly-brace markers stripped.
        - milestone_ids: List of milestone IDs found in {…} markers.
        - page_labels: List of page labels found in [Na]/[Nb] markers (not line markers).
    """
    raw = txt_path.read_text(encoding="utf-8")

    milestone_ids = _MILESTONE_RE.findall(raw)
    page_labels = _PAGE_MARKER_RE.findall(raw)

    # Apply variant corrections: the orig form appears inline before the marker,
    # and the marker {orig,corr} or (orig,corr) says to replace orig with corr.
    # We find orig immediately before the marker and replace orig+marker with corr.
    def _apply_variants(text: str, pattern: re.Pattern[str]) -> str:
        result = text
        for m in reversed(list(pattern.finditer(text))):
            orig = m.group(1)
            corr = m.group(2)
            marker_start = m.start()
            marker_end = m.end()
            # Look for orig immediately before the marker
            orig_start = marker_start - len(orig)
            if orig_start >= 0 and result[orig_start:marker_start] == orig:
                result = result[:orig_start] + corr + result[marker_end:]
            else:
                # orig not found before marker; just replace marker with corr
                result = result[:marker_start] + corr + result[marker_end:]
        return result

    plain_text = _apply_variants(raw, _VARIANT_CURLY_RE)
    plain_text = _apply_variants(plain_text, _VARIANT_PAREN_RE)
    # Remove section milestone markers {D1} etc.
    plain_text = _MILESTONE_RE.sub("", plain_text)
    # Remove bracket markers [1b], [1b.1] etc.
    plain_text = _BRACKET_MARKER_RE.sub("", plain_text)
    # Remove # deletion markers (stripped during TEI generation)
    plain_text = plain_text.replace("#", "")
    plain_text = plain_text.strip()

    return plain_text, milestone_ids, page_labels


def find_source_txt(ie_dir: Path, ve_id: str) -> Path | None:
    """Find the source .txt file for a given VE ID."""
    sources_dir = ie_dir / "sources" / ve_id
    if not sources_dir.exists():
        return None
    txt_files = sorted(sources_dir.glob("*.txt"))
    return txt_files[0] if txt_files else None


def verify_volume(
    xml_path: Path,
    txt_path: Path,
) -> tuple[bool, list[str], list[str]]:
    """Verify a single volume's TEI conversion against its source .txt.

    Returns:
        (passed, issues, warnings) — passed is True if all checks pass,
        issues is a list of hard failures, warnings is a list of known edge cases.
    """
    issues: list[str] = []
    warnings: list[str] = []

    # Parse source .txt
    src_text, src_milestones, src_pages = parse_source_txt(txt_path)

    # Parse TEI XML via tei_to_standoff
    parser = etree.XMLParser(remove_blank_text=True, remove_comments=True, remove_pis=True)
    try:
        tree = etree.parse(str(xml_path), parser)
    except etree.XMLSyntaxError as exc:
        issues.append(f"XML parse error: {exc}")
        return False, issues, []

    root = tree.getroot()
    tei_text, annotations, _source_path = convert_tei_root_to_standoff(root)

    if tei_text is None or annotations is None:
        issues.append("convert_tei_root_to_standoff returned None")
        return False, issues, []

    # ── 1. Compare text content ───────────────────────────────────────────
    # The TEI converter adds double newlines around <pb> page breaks, while
    # the source .txt just has single newlines between markers. Normalize both
    # to single newlines for comparison.
    # Normalize: collapse multiple newlines, strip trailing spaces before newlines,
    # and apply Unicode NFC normalization for Tibetan combining characters.
    def _normalize_text(text: str) -> str:
        text = text.lstrip("\ufeff")
        text = unicodedata.normalize("NFC", text)
        text = re.sub(r" {2,}", " ", text)
        text = re.sub(r" +\n", "\n", text)
        text = re.sub(r"\n{2,}", "\n", text)
        return text.strip()

    tei_normalized = _normalize_text(tei_text)
    src_normalized = _normalize_text(src_text)

    if src_normalized != tei_normalized:
        # Find first divergence point for a helpful message
        min_len = min(len(src_normalized), len(tei_normalized))
        diff_pos = min_len  # default: lengths differ
        for i in range(min_len):
            if src_normalized[i] != tei_normalized[i]:
                diff_pos = i
                break

        context_radius = 30
        src_ctx = src_normalized[max(0, diff_pos - context_radius) : diff_pos + context_radius]
        tei_ctx = tei_normalized[max(0, diff_pos - context_radius) : diff_pos + context_radius]
        issues.append(
            f"Text mismatch at char {diff_pos} "
            f"(src len={len(src_normalized)}, tei len={len(tei_normalized)})\n"
            f"    src: ...{src_ctx!r}...\n"
            f"    tei: ...{tei_ctx!r}..."
        )

    # ── 2. Compare milestones ─────────────────────────────────────────────
    tei_milestones_dict = annotations.get("milestones", {})
    # Filter out synthetic bop:/eop: milestones
    tei_milestone_ids = {
        mid for mid in tei_milestones_dict if not mid.startswith("bop:") and not mid.startswith("eop:")
    }
    src_milestone_set = set(src_milestones)

    missing_in_tei = src_milestone_set - tei_milestone_ids
    extra_in_tei = tei_milestone_ids - src_milestone_set

    if missing_in_tei:
        issues.append(f"Milestones in source but not in TEI output: {sorted(missing_in_tei)}")
    if extra_in_tei:
        issues.append(f"Milestones in TEI output but not in source: {sorted(extra_in_tei)}")

    # ── 3. Compare page labels ────────────────────────────────────────────
    tei_pages = annotations.get("pages", [])
    tei_page_labels = [p.get("pname", "") for p in tei_pages if p.get("pname")]

    src_page_set = set(src_pages)
    tei_page_set = set(tei_page_labels)

    missing_pages = src_page_set - tei_page_set
    extra_pages = tei_page_set - src_page_set

    # Page 1a is often in source but absent from TEI (title page with no content);
    # treat it as a warning, not a failure.
    missing_pages_significant = {p for p in missing_pages if p != "1a"}
    if missing_pages_significant:
        issues.append(f"Pages in source but not in TEI output: {sorted(missing_pages_significant)}")
    elif missing_pages:
        warnings.append("Page 1a in source but not in TEI (known edge case)")
    if extra_pages:
        issues.append(f"Pages in TEI output but not in source: {sorted(extra_pages)}")

    # ── 4. Verify milestone offsets are within text bounds ────────────────
    # Only check section milestones, not synthetic bop:/eop: which can be at
    # edge positions for first/last pages.
    text_len = len(tei_text)
    out_of_bounds = {
        mid: offset
        for mid, offset in tei_milestones_dict.items()
        if not mid.startswith("bop:") and not mid.startswith("eop:") and (offset < 0 or offset > text_len)
    }
    if out_of_bounds:
        issues.append(f"Milestones with out-of-bounds offsets: {out_of_bounds}")

    # ── 5. Verify page offsets don't overlap incorrectly ──────────────────
    for i, page in enumerate(tei_pages):
        cstart = page.get("cstart", 0)
        cend = page.get("cend", 0)
        pname = page.get("pname", str(i))
        if cstart > cend:
            if pname == "1a":
                warnings.append(f"Page 1a: cstart ({cstart}) > cend ({cend}) (known first-page edge case)")
            else:
                issues.append(f"Page {pname}: cstart ({cstart}) > cend ({cend})")
        if cstart < 0 or cend > text_len:
            if pname == "1a":
                warnings.append(f"Page 1a: offset out of bounds ({cstart}-{cend}, text_len={text_len})")
            else:
                issues.append(f"Page {pname}: offset out of bounds ({cstart}-{cend}, text_len={text_len})")

    passed = len(issues) == 0
    return passed, issues, warnings


def main() -> None:
    arg_parser = argparse.ArgumentParser(description="Verify TEI-to-standoff conversion")
    arg_parser.add_argument("--ie-id", required=True, help="Instance Edition ID (e.g., IE1ER199)")
    arg_parser.add_argument("--ie-dir", default=None, help="Path to IE directory")
    arg_parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Base data directory")
    arg_parser.add_argument("--limit", type=int, default=0, help="Max volumes to check (0 = all)")
    arg_parser.add_argument("--verbose", "-v", action="store_true", help="Print details for passing volumes too")
    args = arg_parser.parse_args()

    ie_dir = Path(args.ie_dir) if args.ie_dir else Path(args.data_dir) / args.ie_id

    volumes = discover_volumes(ie_dir)
    if not volumes:
        logger.error("No volumes found in %s", ie_dir)
        return

    if args.limit:
        volumes = volumes[: args.limit]

    total = len(volumes)
    passed_count = 0
    failed_count = 0
    skipped_count = 0

    logger.info("Verifying %d volumes for %s", total, args.ie_id)
    logger.info("=" * 70)

    for idx, (ve_id, xml_path) in enumerate(volumes, start=1):
        txt_path = find_source_txt(ie_dir, ve_id)
        if txt_path is None:
            logger.warning("[%d/%d] %s — SKIP (no source .txt found)", idx, total, ve_id)
            skipped_count += 1
            continue

        passed, issues, vol_warnings = verify_volume(xml_path, txt_path)

        if passed:
            passed_count += 1
            if args.verbose:
                if vol_warnings:
                    logger.info("[%d/%d] %s — PASS (%d warnings)", idx, total, ve_id, len(vol_warnings))
                    for warning in vol_warnings:
                        logger.info("    ⚠ %s", warning)
                else:
                    logger.info("[%d/%d] %s — PASS", idx, total, ve_id)
        else:
            failed_count += 1
            logger.error("[%d/%d] %s — FAIL (%d issues)", idx, total, ve_id, len(issues))
            for issue in issues:
                logger.error("    %s", issue)
            for warning in vol_warnings:
                logger.warning("    ⚠ %s", warning)

    logger.info("=" * 70)
    logger.info(
        "Results: %d passed, %d failed, %d skipped out of %d total",
        passed_count,
        failed_count,
        skipped_count,
        total,
    )

    if failed_count == 0 and skipped_count == 0:
        logger.info("All volumes PASSED verification.")
    elif failed_count == 0:
        logger.info("All checked volumes PASSED (%d skipped).", skipped_count)


if __name__ == "__main__":
    main()
