from __future__ import annotations

from typing import Any

from scripts.dedup.methods.base import BaseDedupMethod, QueryMatch, QueryScope, TextDoc
from scripts.dedup.registry import register_method
from scripts.dedup.text import jaccard, lsh_bands, minhash_values, shingles_from_tokens, tibetan_syllables


class MinhashDatasketchMethod(BaseDedupMethod[set[str]]):
    name = "minhash_datasketch"

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        super().__init__(options)
        self.shingle_size = int(self.options.get("shingle_size", 3))
        self.num_perm = int(self.options.get("num_perm", 128))
        self.bands = int(self.options.get("bands", 20))
        self.rows = int(self.options.get("rows", self.num_perm // self.bands))

    @property
    def supports_pair_score(self) -> bool:
        return True

    @property
    def supports_query(self) -> bool:
        return False

    def fingerprint(self, doc: TextDoc) -> set[str]:
        tokens = tibetan_syllables(doc.text_bo)
        shingles = shingles_from_tokens(tokens, self.shingle_size)
        values = minhash_values(shingles, num_perm=self.num_perm)
        return lsh_bands(values, bands=self.bands, rows=self.rows)

    def pair_score(self, fp_a: set[str], fp_b: set[str]) -> float:
        return jaccard(fp_a, fp_b)

    def query(self, doc: TextDoc, *, top_k: int, scope: QueryScope) -> list[QueryMatch]:
        del doc, top_k, scope
        msg = "minhash_datasketch is pair-score only; benchmark derives closed-set rankings from scores"
        raise NotImplementedError(msg)


@register_method("minhash_datasketch")
def create(options: dict[str, Any]) -> MinhashDatasketchMethod:
    return MinhashDatasketchMethod(options)

