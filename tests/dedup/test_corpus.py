from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from scripts.dedup import corpus
from scripts.dedup.corpus import CorpusConfig, groups_by_field, load_corpus, load_ground_truth_rows

if TYPE_CHECKING:
    import pytest


def _write_csv(path: Path) -> None:
    path.write_text(
        "mw_id,d_id,rkts_id,nlm_id\n"
        "mw1,d1,r1,n1\n"
        "mw2,d1,,n2\n"
        "mw3,d2,r3,\n"
        "mw4,d3,,\n",
        encoding="utf-8",
    )


def test_load_ground_truth_rows_applies_allow_deny_and_subset(tmp_path: Path) -> None:
    csv_path = tmp_path / "ground_truth.csv"
    allowlist = tmp_path / "allow.txt"
    denylist = tmp_path / "deny.txt"
    _write_csv(csv_path)
    allowlist.write_text("mw1\nmw2\nmw3\n", encoding="utf-8")
    denylist.write_text("mw2\n", encoding="utf-8")

    rows = load_ground_truth_rows(
        CorpusConfig(
            csv_path=csv_path,
            filter_in_index=False,
            allowlist_path=allowlist,
            denylist_path=denylist,
            mw_id_subset=frozenset({"mw1", "mw2"}),
        )
    )

    assert list(rows) == ["mw1"]
    assert rows["mw1"].rkts_id == "r1"


def test_load_corpus_can_filter_index_without_loading_text(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "ground_truth.csv"
    _write_csv(csv_path)
    calls: list[bool] = []

    def fake_mget_sources(
        index_name: str,
        mw_ids: list[str],
        *,
        load_source_text: bool,
    ) -> dict[str, dict[str, str]]:
        calls.append(load_source_text)
        assert index_name == "bec_texts"
        assert mw_ids == ["mw1", "mw2", "mw3", "mw4"]
        return {"mw1": {"mw_id": "mw1"}, "mw3": {"mw_id": "mw3", "wa_id_orig": "wa3"}}

    monkeypatch.setattr(corpus, "_mget_sources", fake_mget_sources)

    docs, _rows = load_corpus(CorpusConfig(csv_path=csv_path, load_source_text=False))

    assert calls == [False]
    assert [doc.mw_id for doc in docs] == ["mw1", "mw3"]
    assert docs[0].text_bo == ""
    assert docs[1].wa_id_orig == "wa3"


def test_groups_by_field_only_keeps_positive_groups() -> None:
    docs = [
        corpus.TextDoc("a", "", d_id="g1"),
        corpus.TextDoc("b", "", d_id="g1"),
        corpus.TextDoc("c", "", d_id="g2"),
        corpus.TextDoc("d", "", d_id=None),
    ]

    assert groups_by_field(docs, "d_id") == {"g1": {"a", "b"}}
