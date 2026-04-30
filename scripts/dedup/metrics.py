from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from scripts.dedup.corpus import positive_pairs

if TYPE_CHECKING:
    from scripts.dedup.methods.base import QueryMatch, TextDoc


@dataclass(frozen=True)
class PairScore:
    id_a: str
    id_b: str
    score: float
    is_positive: bool
    group_id: str | None = None


@dataclass(frozen=True)
class ThresholdMetrics:
    threshold: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0


def threshold_sweep(pair_scores: list[PairScore]) -> list[ThresholdMetrics]:
    total_positives = sum(1 for pair in pair_scores if pair.is_positive)
    sorted_pairs = sorted(
        (pair for pair in pair_scores if pair.score > 0.0),
        key=lambda pair: pair.score,
        reverse=True,
    )

    results: list[ThresholdMetrics] = []
    tp = 0
    fp = 0
    index = 0
    while index < len(sorted_pairs):
        threshold = sorted_pairs[index].score
        while index < len(sorted_pairs) and sorted_pairs[index].score == threshold:
            if sorted_pairs[index].is_positive:
                tp += 1
            else:
                fp += 1
            index += 1
        fn = total_positives - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / total_positives if total_positives else 0.0
        results.append(
            ThresholdMetrics(
                threshold=threshold,
                precision=precision,
                recall=recall,
                f1=_f1(precision, recall),
                tp=tp,
                fp=fp,
                fn=fn,
            )
        )
    return results


def pr_auc(sweep: list[ThresholdMetrics]) -> float:
    if not sweep:
        return 0.0
    points = sorted(((metric.recall, metric.precision) for metric in sweep), key=lambda item: item[0])
    area = 0.0
    prev_recall = 0.0
    prev_precision = points[0][1]
    for recall, precision in points:
        area += (recall - prev_recall) * ((precision + prev_precision) / 2)
        prev_recall = recall
        prev_precision = precision
    return area


def best_f1(sweep: list[ThresholdMetrics]) -> ThresholdMetrics:
    if not sweep:
        return ThresholdMetrics(threshold=0.0, precision=0.0, recall=0.0, f1=0.0, tp=0, fp=0, fn=0)
    return max(sweep, key=lambda metric: (metric.f1, metric.recall, metric.precision))


def per_group_recall(pair_scores: list[PairScore]) -> dict[str, float]:
    positives_by_group: dict[str, list[PairScore]] = {}
    for pair in pair_scores:
        if pair.is_positive and pair.group_id:
            positives_by_group.setdefault(pair.group_id, []).append(pair)

    sweep = threshold_sweep(pair_scores)
    if not sweep:
        return dict.fromkeys(positives_by_group, 0.0)

    best = best_f1(sweep)
    return {
        group_id: sum(1 for pair in pairs if pair.score >= best.threshold) / len(pairs)
        for group_id, pairs in positives_by_group.items()
        if pairs
    }


def pair_scores_from_score_map(
    docs: list[TextDoc],
    scores: dict[tuple[str, str], float],
    groups: dict[str, set[str]],
    *,
    missing_score: float = 0.0,
) -> list[PairScore]:
    positives = positive_pairs(groups)
    group_by_pair = {
        pair: group_id
        for group_id, group_ids in groups.items()
        for pair in positive_pairs({group_id: group_ids})
    }
    doc_ids = sorted(doc.mw_id for doc in docs)
    pairs: list[PairScore] = []
    for left_idx, left_id in enumerate(doc_ids):
        for right_id in doc_ids[left_idx + 1 :]:
            key = (left_id, right_id)
            pairs.append(
                PairScore(
                    id_a=left_id,
                    id_b=right_id,
                    score=scores.get(key, missing_score),
                    is_positive=key in positives,
                    group_id=group_by_pair.get(key),
                )
            )
    return pairs


def recall_at_k(
    docs: list[TextDoc],
    query_results: dict[str, list[QueryMatch]],
    groups: dict[str, set[str]],
    *,
    ks: list[int],
) -> dict[str, float]:
    by_doc = {doc.mw_id: doc for doc in docs}
    result: dict[str, float] = {}

    for k in ks:
        recalls: list[float] = []
        for doc in docs:
            if not doc.d_id or doc.d_id not in groups:
                continue
            positives = set(groups[doc.d_id]) - {doc.mw_id}
            if not positives:
                continue
            top_ids = {match.mw_id for match in query_results.get(doc.mw_id, [])[:k]}
            # Open-set queries may return docs outside the benchmark CSV; those are ignored for recall.
            top_ids &= set(by_doc)
            recalls.append(len(top_ids & positives) / len(positives))
        result[f"recall@{k}"] = sum(recalls) / len(recalls) if recalls else 0.0
    return result

