from __future__ import annotations

import gzip
import logging
from pathlib import Path

import requests
from rdflib import ConjunctiveGraph, Literal, Namespace, URIRef

logger = logging.getLogger(__name__)

ENTITY_SCORES_URL = "https://eroux.fr/entityScores.ttl.gz"
BDR = Namespace("http://purl.bdrc.io/resource/")
TMP = Namespace("http://purl.bdrc.io/ontology/tmp/")

_CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
_CACHE_FILE = _CACHE_DIR / "entityScores.ttl"


def _download_scores(*, force: bool = False) -> Path:
    """Download and decompress entityScores.ttl.gz, caching locally."""
    if _CACHE_FILE.exists() and not force:
        logger.info("Using cached entity scores: %s", _CACHE_FILE)
        return _CACHE_FILE

    logger.info("Downloading entity scores from %s", ENTITY_SCORES_URL)
    response = requests.get(ENTITY_SCORES_URL, timeout=120)
    response.raise_for_status()

    decompressed = gzip.decompress(response.content)

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _CACHE_FILE.write_bytes(decompressed)
    logger.info("Cached entity scores to %s (%d bytes)", _CACHE_FILE, len(decompressed))

    return _CACHE_FILE


def load_entity_scores(*, force_download: bool = False) -> dict[str, float]:
    """
    Load entity scores from the BDRC scores file.

    Returns a dict mapping resource local name (e.g. "WA12345") to its score.
    """
    ttl_path = _download_scores(force=force_download)

    logger.info("Parsing entity scores from %s", ttl_path)
    graph = ConjunctiveGraph()
    graph.parse(str(ttl_path), format="turtle")

    scores: dict[str, float] = {}
    bdr_prefix = str(BDR)

    for subject, _predicate, obj in graph.triples((None, TMP.entityScore, None)):
        if not isinstance(subject, URIRef):
            continue
        subject_str = str(subject)
        if not subject_str.startswith(bdr_prefix):
            continue

        local_name = subject_str[len(bdr_prefix) :]

        if isinstance(obj, Literal):
            try:
                scores[local_name] = float(obj)
            except (ValueError, TypeError):
                logger.warning("Non-numeric entity score for %s: %s", local_name, obj)

    logger.info("Loaded %d entity scores", len(scores))
    return scores
