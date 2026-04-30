from __future__ import annotations

from scripts.dedup.methods.base import QueryMatch, TextDoc
from scripts.dedup.metrics import (
    PairScore,
    best_f1,
    pair_scores_from_score_map,
    per_group_recall,
    recall_at_k,
    threshold_sweep,
)


def test_threshold_sweep_ignores_zero_scores() -> None:
    sweep = threshold_sweep(
        [
            PairScore("a", "b", 0.0, is_positive=True, group_id="g1"),
            PairScore("a", "c", -1.0, is_positive=False),
            PairScore("b", "c", 0.8, is_positive=True, group_id="g1"),
        ]
    )

    assert [metric.threshold for metric in sweep] == [0.8]
    assert best_f1(sweep).recall == 0.5


def test_per_group_recall_returns_zero_without_positive_scores() -> None:
    assert per_group_recall([PairScore("a", "b", 0.0, is_positive=True, group_id="g1")]) == {"g1": 0.0}


def test_pair_scores_from_score_map_uses_missing_score() -> None:
    docs = [
        TextDoc(mw_id="a", text_bo="", d_id="g1"),
        TextDoc(mw_id="b", text_bo="", d_id="g1"),
        TextDoc(mw_id="c", text_bo="", d_id="g2"),
    ]

    pairs = pair_scores_from_score_map(
        docs,
        {("a", "b"): 0.9},
        {"g1": {"a", "b"}},
        missing_score=-1.0,
    )

    by_id = {(pair.id_a, pair.id_b): pair for pair in pairs}
    assert by_id[("a", "b")].is_positive
    assert by_id[("a", "b")].score == 0.9
    assert by_id[("a", "c")].score == -1.0


def test_recall_at_k_ignores_results_outside_benchmark() -> None:
    docs = [
        TextDoc(mw_id="a", text_bo="", d_id="g1"),
        TextDoc(mw_id="b", text_bo="", d_id="g1"),
        TextDoc(mw_id="c", text_bo="", d_id="g1"),
    ]
    rankings = {"a": [QueryMatch("external", 10.0), QueryMatch("b", 9.0)]}

    assert recall_at_k(docs, rankings, {"g1": {"a", "b", "c"}}, ks=[1, 2]) == {
        "recall@1": 0.0,
        "recall@2": 1 / 6,
    }
