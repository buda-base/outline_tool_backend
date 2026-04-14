"""
Benchmark deduplication: datasketch LSH bands vs OpenSearch builtin min_hash.

The builtin approach uses the OpenSearch min_hash token filter (hash_count=1,
bucket_count=512, 5-gram shingles) queried via termvectors + terms with
minimum_should_match. See doc/README_OSmin_hash.md for details.

Usage:
    python -m scripts.benchmark_dedup [--sample-size 100] [--min-text-length 200]
                                      [--builtin-min-match-pct 10]

Requires the OpenSearch environment variables from .env.
"""

import argparse
import logging
import re
import time
from typing import Any

from datasketch import MinHash

from api.config import opensearch_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

TEXT_INDEX_NAME = "bec_texts"
NUM_PERM = 128
LSH_BANDS = 20
LSH_ROWS = NUM_PERM // LSH_BANDS
SHINGLE_SIZE = 3
TSHEG_PATTERN = re.compile(r"[་།]+")

MAX_CANDIDATES = 20
BUILTIN_BUCKET_COUNT = 512
BUILTIN_DEFAULT_MIN_MATCH_PCT = 10


def _tibetan_shingles(text: str, shingle_size: int = SHINGLE_SIZE) -> set[str]:
    """Split Tibetan text into overlapping syllable n-grams (shingles)."""
    syllables = [s for s in TSHEG_PATTERN.split(text) if s.strip()]
    if len(syllables) < shingle_size:
        return {"_".join(syllables)} if syllables else set()
    return {"_".join(syllables[i : i + shingle_size]) for i in range(len(syllables) - shingle_size + 1)}


def compute_jaccard(text_a: str, text_b: str) -> float:
    """Compute exact Jaccard similarity between two texts using shingles."""
    shingles_a = _tibetan_shingles(text_a)
    shingles_b = _tibetan_shingles(text_b)
    if not shingles_a or not shingles_b:
        return 0.0
    intersection = len(shingles_a & shingles_b)
    union = len(shingles_a | shingles_b)
    return intersection / union if union > 0 else 0.0


def compute_minhash_jaccard(text_a: str, text_b: str) -> float:
    """Estimate Jaccard similarity using MinHash."""
    mh_a = MinHash(num_perm=NUM_PERM)
    for s in _tibetan_shingles(text_a):
        mh_a.update(s.encode("utf-8"))

    mh_b = MinHash(num_perm=NUM_PERM)
    for s in _tibetan_shingles(text_b):
        mh_b.update(s.encode("utf-8"))

    return mh_a.jaccard(mh_b)


def sample_texts(sample_size: int, min_text_length: int) -> list[dict[str, Any]]:
    """Fetch a random sample of texts from the index."""
    query: dict[str, Any] = {
        "size": sample_size,
        "query": {
            "function_score": {
                "query": {"range": {"text_length": {"gte": min_text_length}}},
                "random_score": {"seed": 42, "field": "_seq_no"},
            }
        },
        "_source": ["volume_id", "wa_id_orig", "title_bo", "text_bo", "text_length", "minhash_lsh"],
    }
    response = opensearch_client.search(index=TEXT_INDEX_NAME, body=query)
    return [{"id": hit["_id"], **hit["_source"]} for hit in response["hits"]["hits"]]


def query_datasketch(doc: dict[str, Any]) -> dict[str, Any]:
    """Query for candidates using the datasketch minhash_lsh field."""
    bands = doc.get("minhash_lsh", [])
    if not bands:
        return {"candidates": [], "latency_ms": 0}

    query: dict[str, Any] = {
        "size": MAX_CANDIDATES + 1,
        "_source": ["volume_id", "wa_id_orig", "title_bo", "text_bo", "text_length"],
        "query": {
            "bool": {
                "must": [{"terms": {"minhash_lsh": bands}}],
                "must_not": [{"term": {"_id": doc["id"]}}],
            }
        },
    }

    t0 = time.monotonic()
    response = opensearch_client.search(index=TEXT_INDEX_NAME, body=query)
    latency_ms = (time.monotonic() - t0) * 1000

    candidates = [
        {"id": hit["_id"], "score": hit["_score"], **hit["_source"]}
        for hit in response["hits"]["hits"][:MAX_CANDIDATES]
    ]
    return {"candidates": candidates, "latency_ms": latency_ms}


def get_builtin_hashes(doc_id: str) -> list[str]:
    """Fetch all stored min_hash tokens for a document via _termvectors.

    With hash_count=1, bucket_count=512 config, this returns ~512 tokens.
    """
    response = opensearch_client.termvectors(
        index=TEXT_INDEX_NAME,
        id=doc_id,
        fields=["text_bo.min_hash_lenient"],
    )
    terms_dict = response.get("term_vectors", {}).get("text_bo.min_hash_lenient", {}).get("terms", {})
    return list(terms_dict.keys())


def query_builtin(
    doc_id: str,
    hashes: list[str],
    min_match: int,
) -> dict[str, Any]:
    """Query for candidates using ALL stored min_hash tokens.

    With the new config (hash_count=1, bucket_count=512), all ~512 tokens
    are used as `should` clauses, and `minimum_should_match` controls the
    similarity threshold: min_match = target_similarity% × 512.

    See doc/README_OSmin_hash.md for details.
    """
    if not hashes:
        return {"candidates": [], "latency_ms": 0, "tokens_used": 0}

    should_clauses = [{"term": {"text_bo.min_hash_lenient": h}} for h in hashes]

    query: dict[str, Any] = {
        "size": MAX_CANDIDATES + 1,
        "_source": ["volume_id", "wa_id_orig", "title_bo", "text_bo", "text_length"],
        "query": {
            "bool": {
                "should": should_clauses,
                "minimum_should_match": min_match,
                "must_not": [{"term": {"_id": doc_id}}],
            }
        },
    }

    t0 = time.monotonic()
    response = opensearch_client.search(index=TEXT_INDEX_NAME, body=query)
    latency_ms = (time.monotonic() - t0) * 1000

    candidates = [
        {"id": hit["_id"], "score": hit["_score"], **hit["_source"]}
        for hit in response["hits"]["hits"][:MAX_CANDIDATES]
    ]
    return {"candidates": candidates, "latency_ms": latency_ms, "tokens_used": len(hashes)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark dedup: datasketch vs builtin min_hash")
    parser.add_argument("--sample-size", type=int, default=100, help="Number of texts to test")
    parser.add_argument("--min-text-length", type=int, default=200, help="Minimum text length to sample")
    parser.add_argument(
        "--builtin-min-match-pct",
        type=int,
        default=BUILTIN_DEFAULT_MIN_MATCH_PCT,
        help="Minimum similarity %% for builtin (applied to 512 tokens). Default: 10%%",
    )
    args = parser.parse_args()

    min_match = max(1, int(args.builtin_min_match_pct / 100 * BUILTIN_BUCKET_COUNT))
    logger.info(
        "Builtin config: %d%% threshold → minimum_should_match=%d (of %d tokens)",
        args.builtin_min_match_pct, min_match, BUILTIN_BUCKET_COUNT,
    )

    logger.info("Sampling %d texts with min length %d...", args.sample_size, args.min_text_length)
    sample = sample_texts(args.sample_size, args.min_text_length)
    logger.info("Got %d sample texts", len(sample))

    ds_latencies: list[float] = []
    ds_counts: list[int] = []
    ds_jaccards: list[float] = []
    ds_mh_jaccards: list[float] = []
    ds_with_cands = 0

    bi_latencies: list[float] = []
    bi_counts: list[int] = []
    bi_jaccards: list[float] = []
    bi_with_cands = 0
    bi_total_hashes: list[int] = []
    bi_errors = 0

    overlaps: list[float] = []
    ds_examples: list[tuple[dict[str, Any], dict[str, Any], float]] = []
    bi_examples: list[tuple[dict[str, Any], dict[str, Any], float]] = []

    for i, doc in enumerate(sample):
        # --- Datasketch ---
        ds_result = query_datasketch(doc)
        ds_latencies.append(ds_result["latency_ms"])
        ds_counts.append(len(ds_result["candidates"]))
        if ds_result["candidates"]:
            ds_with_cands += 1

        # --- Builtin (termvectors + terms with minimum_should_match) ---
        hashes = get_builtin_hashes(doc["id"])
        bi_total_hashes.append(len(hashes))
        try:
            bi_result = query_builtin(doc["id"], hashes, min_match)
            bi_latencies.append(bi_result["latency_ms"])
            bi_counts.append(len(bi_result["candidates"]))
            if bi_result["candidates"]:
                bi_with_cands += 1
        except Exception:
            bi_errors += 1
            bi_result = {"candidates": []}
            bi_latencies.append(0)
            bi_counts.append(0)

        # --- Overlap ---
        ds_ids = {c["id"] for c in ds_result["candidates"]}
        bi_ids = {c["id"] for c in bi_result["candidates"]}
        if ds_ids or bi_ids:
            union = ds_ids | bi_ids
            intersection = ds_ids & bi_ids
            overlaps.append(len(intersection) / len(union) if union else 0.0)

        # --- Quality ---
        source_text = doc["text_bo"]
        for cand in ds_result["candidates"][:5]:
            exact_j = compute_jaccard(source_text, cand["text_bo"])
            mh_j = compute_minhash_jaccard(source_text, cand["text_bo"])
            ds_jaccards.append(exact_j)
            ds_mh_jaccards.append(mh_j)
            if exact_j > 0.1:
                ds_examples.append((doc, cand, exact_j))

        for cand in bi_result["candidates"][:5]:
            exact_j = compute_jaccard(source_text, cand["text_bo"])
            bi_jaccards.append(exact_j)
            if exact_j > 0.1:
                bi_examples.append((doc, cand, exact_j))

        if (i + 1) % 10 == 0:
            logger.info(
                "[%d/%d] ds: %.0fms/%.1f cands | bi: %.0fms/%.1f cands%s",
                i + 1,
                len(sample),
                _mean(ds_latencies),
                _mean(ds_counts),
                _mean(bi_latencies),
                _mean(bi_counts),
                f" ({bi_errors} errors)" if bi_errors else "",
            )

    doc_count = opensearch_client.count(index=TEXT_INDEX_NAME)["count"]

    print("\n" + "=" * 70)
    print("DEDUP BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Sample: {len(sample)} texts (min length: {args.min_text_length} chars)")
    print(f"Index: {TEXT_INDEX_NAME} ({doc_count} docs)")
    print(
        f"Builtin: all ~{_mean(bi_total_hashes):.0f} tokens, "
        f"min_should_match={min_match} ({args.builtin_min_match_pct}% of {BUILTIN_BUCKET_COUNT})"
    )
    if bi_errors:
        print(f"Builtin errors: {bi_errors}/{len(sample)} (likely maxClauseCount)")
    print()

    print("--- LATENCY (ms) ---")
    print(f"  {'':30s} {'mean':>6s}  {'median':>6s}  {'p95':>6s}  {'max':>6s}")
    print(
        f"  {'Datasketch (20 LSH bands)':30s} "
        f"{_mean(ds_latencies):6.0f}  {_median(ds_latencies):6.0f}  "
        f"{_percentile(ds_latencies, 0.95):6.0f}  {max(ds_latencies):6.0f}"
    )
    print(
        f"  {'Builtin (all tokens+msm)':30s} "
        f"{_mean(bi_latencies):6.0f}  {_median(bi_latencies):6.0f}  "
        f"{_percentile(bi_latencies, 0.95):6.0f}  {max(bi_latencies):6.0f}"
    )
    print()

    print("--- CANDIDATE COUNTS ---")
    print(f"  {'':30s} {'mean':>6s}  {'median':>6s}  {'max':>6s}  {'with_cands':>10s}")
    print(
        f"  {'Datasketch':30s} "
        f"{_mean(ds_counts):6.1f}  {_median(ds_counts):6.0f}  "
        f"{max(ds_counts):6d}  {ds_with_cands:>10d}/{len(sample)}"
    )
    print(
        f"  {'Builtin':30s} "
        f"{_mean(bi_counts):6.1f}  {_median(bi_counts):6.0f}  "
        f"{max(bi_counts):6d}  {bi_with_cands:>10d}/{len(sample)}"
    )
    print()

    print("--- OVERLAP (datasketch vs builtin candidates) ---")
    if overlaps:
        print(f"  Jaccard overlap: mean={_mean(overlaps):.2%}  median={_median(overlaps):.2%}")
    else:
        print("  No overlaps to compute (no candidates found)")
    print()

    def _print_quality(label: str, jaccards: list[float]) -> None:
        if jaccards:
            high = sum(1 for j in jaccards if j >= 0.5)
            medium = sum(1 for j in jaccards if 0.2 <= j < 0.5)
            low = sum(1 for j in jaccards if j < 0.2)
            print(
                f"  {label:30s} mean={_mean(jaccards):.3f}  "
                f"median={_median(jaccards):.3f}  max={max(jaccards):.3f}  n={len(jaccards)}"
            )
            print(f"  {'':30s} strong(>=0.5)={high}  partial(0.2-0.5)={medium}  weak(<0.2)={low}")
        else:
            print(f"  {label:30s} no candidates found")

    print("--- RESULT QUALITY (actual Jaccard of top-5 candidates) ---")
    _print_quality("Datasketch", ds_jaccards)
    if ds_mh_jaccards:
        print(
            f"  {'Datasketch (MH estimate)':30s} mean={_mean(ds_mh_jaccards):.3f}  "
            f"median={_median(ds_mh_jaccards):.3f}  max={max(ds_mh_jaccards):.3f}"
        )
    _print_quality("Builtin", bi_jaccards)
    print()

    def _print_examples(label: str, examples_list: list[tuple[dict[str, Any], dict[str, Any], float]]) -> None:
        print(f"--- EXAMPLE MATCHES: {label} (top Jaccard) ---")
        examples_list.sort(key=lambda x: x[2], reverse=True)
        for src_doc, cand, j in examples_list[:10]:
            src_title = (src_doc.get("title_bo") or "")[:40]
            cand_title = (cand.get("title_bo") or "")[:40]
            print(f"  J={j:.3f}  src={src_doc['id'][:50]} (wa={src_doc.get('wa_id_orig', '?')})  [{src_title}]")
            print(f"           cand={cand['id'][:50]} (wa={cand.get('wa_id_orig', '?')})  [{cand_title}]")
        if not examples_list:
            print("  No matches with Jaccard > 0.1 found")
        print()

    _print_examples("Datasketch", ds_examples)
    _print_examples("Builtin", bi_examples)

    print("=" * 70)


def _mean(values: list[float | int]) -> float:
    return sum(values) / len(values) if values else 0.0


def _median(values: list[float | int]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    n = len(sorted_values)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_values[mid - 1] + sorted_values[mid]) / 2
    return sorted_values[mid]


def _percentile(values: list[float | int], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = int(len(sorted_values) * pct)
    return sorted_values[min(idx, len(sorted_values) - 1)]


if __name__ == "__main__":
    main()
