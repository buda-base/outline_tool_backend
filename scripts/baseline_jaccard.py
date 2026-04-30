"""
Baseline Jaccard analysis for known duplicate texts across OCR versions.

Takes groups of texts that are known to be the same work (from nlm_merged.csv,
grouped by d_id) and computes pairwise Jaccard similarity at different shingle
sizes (1-6). This establishes the expected similarity range for true duplicates
under OCR noise, so we can tune the MinHash dedup thresholds.

Also compares the Python shingling (used by datasketch) with the OpenSearch
analyzer's tokenization to see if they produce comparable Jaccard values.

Usage:
    python -m scripts.baseline_jaccard --csv path/to/nlm_merged.csv [--limit 10]
    python -m scripts.baseline_jaccard --groups D4392 D4405 D4387

Requires the OpenSearch environment variables from .env.
"""

import argparse
import csv
import logging
import random
import re
import statistics
import time
from collections import defaultdict
from itertools import combinations
from typing import Any

from rapidfuzz.distance import Levenshtein

from api.config import opensearch_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

TEXT_INDEX_NAME = "bec_texts"
TSHEG_PATTERN = re.compile(r"[་།]+")


def _tibetan_shingles(text: str, shingle_size: int) -> set[str]:
    """Split Tibetan text into overlapping syllable n-grams."""
    syllables = [s for s in TSHEG_PATTERN.split(text) if s.strip()]
    if len(syllables) < shingle_size:
        return {"_".join(syllables)} if syllables else set()
    return {
        "_".join(syllables[i : i + shingle_size])
        for i in range(len(syllables) - shingle_size + 1)
    }


def jaccard(set_a: set[str], set_b: set[str]) -> float:
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0


def char_distance(text_a: str, text_b: str) -> dict[str, Any]:
    """Character-level distance metrics including normalized Levenshtein."""
    len_a, len_b = len(text_a), len(text_b)
    avg_len = (len_a + len_b) / 2 if (len_a + len_b) > 0 else 1
    chars_a, chars_b = set(text_a), set(text_b)
    char_jaccard = len(chars_a & chars_b) / len(chars_a | chars_b) if chars_a | chars_b else 0.0

    lev_dist = Levenshtein.distance(text_a, text_b)
    norm_lev = lev_dist / avg_len

    return {
        "len_a": len_a,
        "len_b": len_b,
        "len_ratio": min(len_a, len_b) / max(len_a, len_b) if max(len_a, len_b) > 0 else 0.0,
        "char_jaccard": char_jaccard,
        "levenshtein": lev_dist,
        "norm_lev": norm_lev,
        "lev_similarity": 1.0 - norm_lev,
    }


MAX_ANALYZE_CHARS = 15000

def get_os_tokens(text: str, analyzer: str = "tibetan-lenient") -> list[str]:
    """Get tokens from OpenSearch's analyzer for a text.

    Truncates to MAX_ANALYZE_CHARS to stay under index.analyze.max_token_count.
    """
    truncated = text[:MAX_ANALYZE_CHARS]
    response = opensearch_client.indices.analyze(
        index=TEXT_INDEX_NAME,
        body={"analyzer": analyzer, "text": truncated},
    )
    return [t["token"] for t in response.get("tokens", [])]


def os_shingles(tokens: list[str], shingle_size: int) -> set[str]:
    """Build shingles from pre-tokenized OpenSearch tokens."""
    if len(tokens) < shingle_size:
        return {"_".join(tokens)} if tokens else set()
    return {
        "_".join(tokens[i : i + shingle_size])
        for i in range(len(tokens) - shingle_size + 1)
    }


def get_os_minhash_tokens(doc_id: str) -> list[str]:
    """Fetch the stored min_hash tokens for a document."""
    response = opensearch_client.termvectors(
        index=TEXT_INDEX_NAME,
        id=doc_id,
        fields=["text_bo.min_hash_lenient"],
    )
    terms = response.get("term_vectors", {}).get("text_bo.min_hash_lenient", {}).get("terms", {})
    return list(terms.keys())


def fetch_texts(mw_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Fetch texts from bec_texts by their mw_id (= doc _id)."""
    if not mw_ids:
        return {}
    response = opensearch_client.mget(
        body={"ids": mw_ids},
        index=TEXT_INDEX_NAME,
    )
    result = {}
    for doc in response["docs"]:
        if doc.get("found"):
            src = doc["_source"]
            full_text = src["text_bo"]
            cstart = src.get("cstart", 0)
            cend = src.get("cend", 0)
            cs_clean = src.get("cstart_clean")
            ce_clean = src.get("cend_clean")

            if cs_clean is not None and ce_clean is not None and cend > cstart:
                trim_start = cs_clean - cstart
                trim_end = len(full_text) - (cend - ce_clean)
                text_clean = full_text[max(0, trim_start):max(0, trim_end)]
            else:
                text_clean = full_text

            result[doc["_id"]] = {
                "text_bo": full_text,
                "text_clean": text_clean,
                "text_length": src.get("text_length", len(full_text)),
                "title_bo": src.get("title_bo", ""),
                "wa_id_orig": src.get("wa_id_orig"),
                "etext_source": src.get("etext_source", "unknown"),
                "minhash_lsh": src.get("minhash_lsh", []),
                "boundary_start": src.get("boundary_start", "unknown"),
                "boundary_end": src.get("boundary_end", "unknown"),
            }
    return result


def load_csv_groups(csv_path: str) -> dict[str, list[str]]:
    """Load nlm_merged.csv and group mw_ids by d_id."""
    groups: dict[str, list[str]] = defaultdict(list)
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            d_id = row.get("d_id", "").strip()
            mw_id = row.get("mw_id", "").strip()
            if d_id and mw_id:
                groups[d_id].append(mw_id)
    # Only keep groups with 2+ texts (need pairs to compare)
    return {k: v for k, v in groups.items() if len(v) >= 2}


def find_cross_engine_texts(d_ids: list[str]) -> dict[str, list[str]]:
    """Find TEI (manual input) texts whose title_bo matches a d_id.

    Returns a mapping of d_id -> list of TEI doc _ids that can be added
    to groups for cross-engine comparison (TEI vs Google Vision OCR).
    """
    response = opensearch_client.search(
        index=TEXT_INDEX_NAME,
        body={
            "size": 500,
            "_source": ["title_bo", "etext_source"],
            "query": {"term": {"etext_source": "tei"}},
        },
    )
    tei_by_did: dict[str, list[str]] = defaultdict(list)
    did_set = set(d_ids)
    for hit in response["hits"]["hits"]:
        title = hit["_source"].get("title_bo", "")
        if title in did_set:
            tei_by_did[title].append(hit["_id"])
    return dict(tei_by_did)


MAX_PAIRS_PER_GROUP = 30


def _pair_type(src_a: str, src_b: str) -> str:
    """Classify source pair: ocr-ocr, tei-ocr, or other."""
    sources = sorted([src_a, src_b])
    if sources == ["google_vision", "google_vision"]:
        return "gv-gv"
    if "tei" in sources and "google_vision" in sources:
        return "tei-gv"
    if "tei" in sources:
        return "tei-other"
    return f"{sources[0]}-{sources[1]}"


def analyze_group(
    d_id: str,
    texts: dict[str, dict[str, Any]],
    mw_ids: list[str],
    shingle_sizes: list[int],
) -> dict[str, Any]:
    """Analyze a group of known-duplicate texts."""
    available = [mid for mid in mw_ids if mid in texts]
    if len(available) < 2:
        return {"d_id": d_id, "available": len(available), "pairs": []}

    all_combos = list(combinations(available, 2))
    if len(all_combos) > MAX_PAIRS_PER_GROUP:
        random.seed(42)
        all_combos = random.sample(all_combos, MAX_PAIRS_PER_GROUP)

    pairs_data = []
    for id_a, id_b in all_combos:
        text_a = texts[id_a]["text_clean"]
        text_b = texts[id_b]["text_clean"]

        src_a = texts[id_a].get("etext_source", "unknown")
        src_b = texts[id_b].get("etext_source", "unknown")
        pair_type = _pair_type(src_a, src_b)

        bnd_a = f"{texts[id_a].get('boundary_start', '?')}/{texts[id_a].get('boundary_end', '?')}"
        bnd_b = f"{texts[id_b].get('boundary_start', '?')}/{texts[id_b].get('boundary_end', '?')}"

        pair_result: dict[str, Any] = {
            "id_a": id_a,
            "id_b": id_b,
            "src_a": src_a,
            "src_b": src_b,
            "pair_type": pair_type,
            "bnd_a": bnd_a,
            "bnd_b": bnd_b,
            "title_a": (texts[id_a].get("title_bo") or "")[:50],
            "title_b": (texts[id_b].get("title_bo") or "")[:50],
            **char_distance(text_a, text_b),
        }

        # Python shingle Jaccard at each size
        for size in shingle_sizes:
            sh_a = _tibetan_shingles(text_a, size)
            sh_b = _tibetan_shingles(text_b, size)
            j = jaccard(sh_a, sh_b)
            pair_result[f"py_j_{size}"] = j
            pair_result[f"py_shingles_{size}"] = (len(sh_a), len(sh_b))

        # OpenSearch analyzer tokens, then shingle Jaccard at each size
        os_tok_a = get_os_tokens(text_a)
        os_tok_b = get_os_tokens(text_b)
        pair_result["os_tokens"] = (len(os_tok_a), len(os_tok_b))

        for size in shingle_sizes:
            os_sh_a = os_shingles(os_tok_a, size)
            os_sh_b = os_shingles(os_tok_b, size)
            j = jaccard(os_sh_a, os_sh_b)
            pair_result[f"os_j_{size}"] = j

        # OS builtin min_hash token overlap (Jaccard of stored hash sets)
        mh_a = set(get_os_minhash_tokens(id_a))
        mh_b = set(get_os_minhash_tokens(id_b))
        pair_result["os_minhash_jaccard"] = jaccard(mh_a, mh_b)
        pair_result["os_minhash_tokens"] = (len(mh_a), len(mh_b))

        # Datasketch LSH band overlap
        bands_a = set(texts[id_a].get("minhash_lsh", []))
        bands_b = set(texts[id_b].get("minhash_lsh", []))
        shared_bands = len(bands_a & bands_b)
        pair_result["ds_shared_bands"] = shared_bands
        pair_result["ds_would_match"] = shared_bands > 0

        pairs_data.append(pair_result)

    return {"d_id": d_id, "available": len(available), "pairs": pairs_data}


def _print_section(
    pairs: list[dict[str, Any]],
    shingle_sizes: list[int],
) -> None:
    """Print stats for a subset of pairs (e.g. all, or filtered by pair_type)."""
    if not pairs:
        print(f"  (no pairs)")
        return

    def _stat_line(lbl: str, vals: list[float]) -> None:
        print(f"  {lbl:20s} mean={statistics.mean(vals):.3f}  "
              f"median={statistics.median(vals):.3f}  "
              f"min={min(vals):.3f}  max={max(vals):.3f}")

    print(f"  Pairs: {len(pairs)}")

    # Character-level
    _stat_line("Length ratio:", [p["len_ratio"] for p in pairs])
    _stat_line("Char Jaccard:", [p["char_jaccard"] for p in pairs])
    _stat_line("Norm Levenshtein:", [p["norm_lev"] for p in pairs])
    _stat_line("Lev similarity:", [p["lev_similarity"] for p in pairs])

    # Jaccard by shingle size
    header = f"  {'shingle':>7s}"
    header += f"  {'py_mean':>8s} {'py_med':>7s} {'py_min':>7s} {'py_max':>7s}"
    header += f"  {'os_mean':>8s} {'os_med':>7s} {'os_min':>7s} {'os_max':>7s}"
    print(header)
    for size in shingle_sizes:
        py_vals = [p[f"py_j_{size}"] for p in pairs]
        os_vals = [p[f"os_j_{size}"] for p in pairs]
        print(
            f"  {size:>7d}"
            f"  {statistics.mean(py_vals):8.3f} {statistics.median(py_vals):7.3f}"
            f" {min(py_vals):7.3f} {max(py_vals):7.3f}"
            f"  {statistics.mean(os_vals):8.3f} {statistics.median(os_vals):7.3f}"
            f" {min(os_vals):7.3f} {max(os_vals):7.3f}"
        )

    # MinHash
    os_mh = [p["os_minhash_jaccard"] for p in pairs]
    ds_sh = [p["ds_shared_bands"] for p in pairs]
    ds_rate = sum(1 for p in pairs if p["ds_would_match"]) / len(pairs)
    _stat_line("OS MinHash Jacc:", os_mh)
    print(f"  {'DS shared bands:':20s} mean={statistics.mean(ds_sh):.1f}  "
          f"median={statistics.median(ds_sh):.1f}  "
          f"min={min(ds_sh)}  max={max(ds_sh)}")
    print(f"  {'DS match rate:':20s} {ds_rate:.1%} of pairs share >= 1 band")

    # Boundary info
    bnd_counts: dict[str, int] = defaultdict(int)
    for p in pairs:
        for side in ("bnd_a", "bnd_b"):
            bnd = p.get(side, "unknown/unknown")
            if "shared_page" in bnd:
                bnd_counts["shared_page"] += 1
            elif "unknown" in bnd:
                bnd_counts["unknown"] += 1
            else:
                bnd_counts["clean"] += 1
    total_sides = len(pairs) * 2
    if bnd_counts.get("shared_page", 0) > 0:
        print(f"  {'Boundary noise:':20s} {bnd_counts['shared_page']}/{total_sides} text sides have shared-page overlap")


def print_results(
    results: list[dict[str, Any]],
    shingle_sizes: list[int],
) -> None:
    all_pairs: list[dict[str, Any]] = []
    for group in results:
        all_pairs.extend(group.get("pairs", []))

    if not all_pairs:
        print("No text pairs found in the index.")
        return

    print("\n" + "=" * 90)
    print("BASELINE JACCARD ANALYSIS — Known Duplicate Texts (OCR variations)")
    print("=" * 90)
    print(f"Groups analyzed: {len(results)}")
    print(f"Text pairs: {len(all_pairs)}")
    print()

    # Identify pair types
    pair_types: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in all_pairs:
        pair_types[p.get("pair_type", "unknown")].append(p)

    # Overall stats
    print("=== ALL PAIRS ===")
    _print_section(all_pairs, shingle_sizes)
    print()

    # Per source-pair-type breakdown
    if len(pair_types) > 1:
        for ptype in sorted(pair_types.keys()):
            print(f"=== {ptype.upper()} PAIRS ===")
            _print_section(pair_types[ptype], shingle_sizes)
            print()

    # Threshold recommendations
    print("--- THRESHOLD IMPLICATIONS ---")
    for size in shingle_sizes:
        py_vals = [p[f"py_j_{size}"] for p in all_pairs]
        min_j = min(py_vals)
        p10 = sorted(py_vals)[max(0, len(py_vals) // 10)]
        print(f"  Shingle {size}: worst-case J={min_j:.3f}, p10={p10:.3f} "
              f"→ min_should_match@512 = {int(min_j * 512)} (worst) / {int(p10 * 512)} (p10)")
    print()

    # Per-pair detail table
    print("--- PER-PAIR DETAIL (sorted by Levenshtein similarity) ---")
    all_pairs.sort(key=lambda p: p.get("lev_similarity", 0), reverse=True)
    print(f"  {'d_id':>8s}  {'type':>6s}  {'id_a':>35s}  {'bnd_a':>20s}  {'id_b':>35s}  {'bnd_b':>20s}  "
          f"{'len_r':>5s}  {'levSim':>6s}  {'nLev':>5s}  "
          f"{'py_1':>5s}  {'py_3':>5s}  {'py_5':>5s}  "
          f"{'os_1':>5s}  {'os_3':>5s}  {'os_5':>5s}  {'os_mh':>5s}  {'ds_b':>4s}")
    print("  " + "-" * 230)

    for p in all_pairs:
        d_id = ""
        for g in results:
            if p in g.get("pairs", []):
                d_id = g["d_id"]
                break
        print(
            f"  {d_id:>8s}  {p.get('pair_type', '?'):>6s}  {p['id_a']:>35s}  {p.get('bnd_a', '?'):>20s}  "
            f"{p['id_b']:>35s}  {p.get('bnd_b', '?'):>20s}  "
            f"{p['len_ratio']:5.2f}  {p.get('lev_similarity', 0):6.3f}  {p.get('norm_lev', 0):5.3f}  "
            f"{p.get('py_j_1', 0):5.3f}  {p.get('py_j_3', 0):5.3f}  {p.get('py_j_5', 0):5.3f}  "
            f"{p.get('os_j_1', 0):5.3f}  {p.get('os_j_3', 0):5.3f}  {p.get('os_j_5', 0):5.3f}  "
            f"{p['os_minhash_jaccard']:5.3f}  {p['ds_shared_bands']:4d}"
        )

    print()
    print("=" * 90)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Baseline Jaccard analysis for known duplicate texts"
    )
    parser.add_argument(
        "--csv",
        help="Path to nlm_merged.csv with columns: mw_id, nlm_id, rkts_id, d_id",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        help="Specific d_id values to analyze (if no CSV)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of groups to analyze (0 = all)",
    )
    parser.add_argument(
        "--shingle-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5, 6],
        help="Shingle sizes to test (default: 1 2 3 4 5 6)",
    )
    parser.add_argument(
        "--no-cross-engine",
        action="store_true",
        help="Skip adding TEI cross-engine texts for comparison",
    )
    args = parser.parse_args()

    if not args.csv and not args.groups:
        parser.error("Provide either --csv or --groups")

    if args.csv:
        logger.info("Loading CSV from %s", args.csv)
        groups = load_csv_groups(args.csv)
        logger.info("Found %d groups with 2+ texts", len(groups))

        if args.groups:
            groups = {k: v for k, v in groups.items() if k in args.groups}
            logger.info("Filtered to %d groups: %s", len(groups), list(groups.keys()))

        if args.limit > 0:
            limited = dict(list(groups.items())[:args.limit])
            groups = limited

        if not args.no_cross_engine:
            logger.info("Looking for cross-engine TEI texts matching d_ids...")
            tei_map = find_cross_engine_texts(list(groups.keys()))
            cross_count = 0
            for d_id, tei_ids in tei_map.items():
                if d_id in groups:
                    groups[d_id].extend(tei_ids)
                    cross_count += len(tei_ids)
                else:
                    groups[d_id] = tei_ids
            if cross_count:
                logger.info("Added %d TEI texts across %d groups for cross-engine comparison",
                            cross_count, len(tei_map))
            else:
                logger.info("No TEI cross-engine texts found for these d_ids")
    else:
        parser.error("--groups without --csv is not yet implemented. Please provide --csv.")
        return

    all_mw_ids = [mid for mids in groups.values() for mid in mids]
    logger.info("Fetching %d texts from bec_texts...", len(all_mw_ids))
    t0 = time.monotonic()

    # Fetch in batches of 100
    all_texts: dict[str, dict[str, Any]] = {}
    for i in range(0, len(all_mw_ids), 100):
        batch = all_mw_ids[i : i + 100]
        batch_texts = fetch_texts(batch)
        all_texts.update(batch_texts)

    fetch_elapsed = time.monotonic() - t0
    logger.info("Fetched %d/%d texts in %.1fs", len(all_texts), len(all_mw_ids), fetch_elapsed)

    missing = set(all_mw_ids) - set(all_texts.keys())
    if missing:
        logger.warning("%d mw_ids not found in bec_texts (first 10): %s", len(missing), list(missing)[:10])

    results = []
    for i, (d_id, mw_ids) in enumerate(groups.items()):
        logger.info("[%d/%d] Analyzing group %s (%d texts)...", i + 1, len(groups), d_id, len(mw_ids))
        result = analyze_group(d_id, all_texts, mw_ids, args.shingle_sizes)
        results.append(result)

    print_results(results, args.shingle_sizes)


if __name__ == "__main__":
    main()
