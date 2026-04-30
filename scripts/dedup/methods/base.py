from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, TypeVar

Fingerprint = TypeVar("Fingerprint")


@dataclass(frozen=True)
class TextDoc:
    """Small projection of a bec_texts document used by benchmarks."""

    mw_id: str
    text_bo: str
    d_id: str | None = None
    rkts_id: str | None = None
    wa_id_orig: str | None = None
    title_bo: str | None = None
    etext_source: str | None = None
    text_length: int = 0


@dataclass(frozen=True)
class QueryScope:
    """Scope a method should search."""

    kind: str
    mw_ids: frozenset[str] = field(default_factory=frozenset)
    index: str = "bec_texts"

    @classmethod
    def closed_set(cls, mw_ids: set[str] | frozenset[str]) -> QueryScope:
        return cls(kind="closed_set", mw_ids=frozenset(mw_ids))

    @classmethod
    def open_set(cls, index: str = "bec_texts") -> QueryScope:
        return cls(kind="open_set", index=index)


@dataclass(frozen=True)
class QueryMatch:
    mw_id: str
    score: float


class DedupMethod(Protocol, Generic[Fingerprint]):  # noqa: UP046
    """Common interface for benchmarkable deduplication methods.

    A method may be pairwise/fingerprint-based, query/index-based, or both.
    The benchmark harness derives the missing operation where practical.
    """

    name: str
    options: dict[str, Any]

    @property
    def supports_pair_score(self) -> bool:
        ...

    @property
    def supports_query(self) -> bool:
        ...

    @property
    def requires_source_text(self) -> bool:
        ...

    def fingerprint(self, doc: TextDoc) -> Fingerprint:
        ...

    def pair_score(self, fp_a: Fingerprint, fp_b: Fingerprint) -> float:
        ...

    def query(self, doc: TextDoc, *, top_k: int, scope: QueryScope) -> list[QueryMatch]:
        ...


class BaseDedupMethod(Generic[Fingerprint]):  # noqa: UP046
    """Base class with explicit unsupported-operation errors."""

    name = "base"

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        self.options = options or {}

    @property
    def supports_pair_score(self) -> bool:
        return False

    @property
    def supports_query(self) -> bool:
        return False

    @property
    def requires_source_text(self) -> bool:
        return True

    def fingerprint(self, doc: TextDoc) -> Fingerprint:
        msg = f"{self.name} does not implement fingerprint()"
        raise NotImplementedError(msg)

    def pair_score(self, fp_a: Fingerprint, fp_b: Fingerprint) -> float:
        msg = f"{self.name} does not implement pair_score()"
        raise NotImplementedError(msg)

    def query(self, doc: TextDoc, *, top_k: int, scope: QueryScope) -> list[QueryMatch]:
        msg = f"{self.name} does not implement query()"
        raise NotImplementedError(msg)

