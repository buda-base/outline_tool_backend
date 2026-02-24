from __future__ import annotations

import logging
from pathlib import Path

import pyewts
from rdflib import ConjunctiveGraph, Literal, Namespace, URIRef

from api.models import ParsedRecord

logger = logging.getLogger(__name__)

ADM = Namespace("http://purl.bdrc.io/ontology/admin/")
BDA = Namespace("http://purl.bdrc.io/admindata/")
BDO = Namespace("http://purl.bdrc.io/ontology/core/")
BDR = Namespace("http://purl.bdrc.io/resource/")
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")

_AUTHOR_ROLES = {BDR.R0ER0011, BDR.R0ER0014, BDR.R0ER0019, BDR.R0ER0025}
_PRIORITY_AUTHOR_ROLE = BDR.R0ER0014

EWTS_CONVERTER = pyewts.pyewts()


def _ewts_to_unicode(ewts_text: str) -> str:
    """Convert EWTS transliteration to Tibetan Unicode."""
    return EWTS_CONVERTER.toUnicode(ewts_text)


def _extract_label(graph: ConjunctiveGraph, subject: URIRef, predicate: URIRef) -> str | None:
    """Extract a label, preferring @bo over @bo-x-ewts (converted)."""
    bo_direct: str | None = None
    bo_ewts: str | None = None

    for obj in graph.objects(subject, predicate):
        if not isinstance(obj, Literal):
            continue
        lang = obj.language
        if lang == "bo":
            bo_direct = str(obj)
        elif lang == "bo-x-ewts":
            bo_ewts = str(obj)

    if bo_direct is not None:
        return bo_direct
    if bo_ewts is not None:
        return _ewts_to_unicode(bo_ewts)
    return None


def _extract_labels(graph: ConjunctiveGraph, subject: URIRef, predicate: URIRef) -> list[str]:
    """Extract all labels for a predicate, preferring @bo over @bo-x-ewts."""
    bo_direct: list[str] = []
    bo_ewts: list[str] = []

    for obj in graph.objects(subject, predicate):
        if not isinstance(obj, Literal):
            continue
        lang = obj.language
        if lang == "bo":
            bo_direct.append(str(obj))
        elif lang == "bo-x-ewts":
            bo_ewts.append(str(obj))

    return bo_direct + [_ewts_to_unicode(e) for e in bo_ewts]


def _extract_authors(graph: ConjunctiveGraph, subject: URIRef) -> list[str]:
    """Extract author person IDs from :creator nodes.

    Traverses :creator blank/named nodes, checks :role against author roles.
    If any creator has role R0ER0014 (commentator), only those are returned.
    Otherwise all matching authors are returned.
    Returns local names (P...) of the :agent URIs.
    """
    bdr_prefix = str(BDR)
    priority_authors: list[str] = []
    other_authors: list[str] = []

    for creator_node in graph.objects(subject, BDO.creator):
        role = None
        agent = None
        for r in graph.objects(creator_node, BDO.role):
            if r in _AUTHOR_ROLES:
                role = r
                break
        if role is None:
            continue
        for a in graph.objects(creator_node, BDO.agent):
            if isinstance(a, URIRef) and str(a).startswith(bdr_prefix):
                agent = str(a)[len(bdr_prefix) :]
                break
        if agent is None:
            continue

        if role == _PRIORITY_AUTHOR_ROLE:
            priority_authors.append(agent)
        else:
            other_authors.append(agent)

    return priority_authors or other_authors


def _detect_type(record_id: str) -> str:
    """Detect record type from ID prefix."""
    if record_id.startswith("W"):
        return "work"
    if record_id.startswith("P"):
        return "person"
    return "unknown"


def parse_trig_file(file_path: Path) -> ParsedRecord | None:
    """
    Parse a single .trig file and return a ParsedRecord.

    Returns None if the file cannot be parsed.
    """
    record_id = file_path.stem

    try:
        graph = ConjunctiveGraph()
        graph.parse(str(file_path), format="trig")
    except Exception:
        logger.exception("Failed to parse trig file: %s", file_path)
        return None

    admin_subject = BDA[record_id]
    resource_subject = BDR[record_id]

    # --- status ---
    status_values = list(graph.objects(admin_subject, ADM.status))
    is_released = BDA.StatusReleased in status_values

    # --- replaceWith ---
    replacement_id: str | None = None
    for obj in graph.objects(admin_subject, ADM.replaceWith):
        if isinstance(obj, URIRef) and str(obj).startswith(str(BDR)):
            replacement_id = str(obj)[len(str(BDR)) :]
            break

    # --- labels ---
    pref_label_bo = _extract_label(graph, resource_subject, SKOS.prefLabel)
    alt_label_bo = _extract_labels(graph, resource_subject, SKOS.altLabel)

    record_type = _detect_type(record_id)

    # --- authors (works only) ---
    authors: list[str] = []
    if record_type == "work":
        authors = _extract_authors(graph, resource_subject)

    return ParsedRecord(
        id=record_id,
        type=record_type,
        is_released=is_released,
        replacement_id=replacement_id,
        pref_label_bo=pref_label_bo,
        alt_label_bo=alt_label_bo,
        authors=authors,
    )
