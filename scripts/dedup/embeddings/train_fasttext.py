from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from api.config import opensearch_client
from scripts.dedup.embeddings.encode import DEFAULT_ANALYZER, analyze_tokens

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

TEXT_INDEX_NAME = "bec_texts"
DEFAULT_CORPUS = Path("data/fasttext/corpus.txt")
DEFAULT_OUTPUT_DIR = Path("data/fasttext")
SCROLL_SIZE = 100
SCROLL_TIMEOUT = "5m"


def _iter_bocorpus() -> Iterable[str]:
    try:
        from datasets import load_dataset  # noqa: PLC0415
    except ImportError as exc:
        logger.warning("datasets is not installed; skipping BoCorpus (%s)", exc)
        return

    try:
        dataset = load_dataset("openpecha/BoCorpus", split="train")
    except Exception as exc:
        logger.warning("Could not load openpecha/BoCorpus; falling back to OCR only (%s)", exc)
        return

    for row in dataset:
        if isinstance(row, dict):
            text = row.get("text") or row.get("content") or row.get("bo")
            if isinstance(text, str) and text.strip():
                yield text


def _iter_bec_texts(index_name: str, limit: int) -> Iterable[str]:
    body: dict[str, Any] = {
        "query": {"exists": {"field": "text_bo"}},
        "_source": ["text_bo"],
    }
    response = opensearch_client.search(
        index=index_name,
        body=body,
        size=SCROLL_SIZE,
        scroll=SCROLL_TIMEOUT,
    )
    scroll_id = response.get("_scroll_id")
    emitted = 0
    try:
        while True:
            hits = response.get("hits", {}).get("hits", [])
            if not hits:
                break
            for hit in hits:
                text = hit.get("_source", {}).get("text_bo", "")
                if text.strip():
                    yield text
                    emitted += 1
                    if limit > 0 and emitted >= limit:
                        return
            response = opensearch_client.scroll(scroll_id=scroll_id, scroll=SCROLL_TIMEOUT)
            scroll_id = response.get("_scroll_id")
    finally:
        if scroll_id:
            opensearch_client.clear_scroll(scroll_id=scroll_id)


def build_corpus(
    *,
    output_path: Path,
    analyzer: str,
    index_name: str,
    limit: int,
    include_bocorpus: bool,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as fh:
        sources: list[Iterable[str]] = []
        if include_bocorpus:
            sources.append(_iter_bocorpus())
        sources.append(_iter_bec_texts(index_name, limit))

        for source in sources:
            for text in source:
                tokens = analyze_tokens(text, analyzer=analyzer, index=index_name)
                if tokens:
                    fh.write(" ".join(tokens) + "\n")
                    count += 1
                if count and count % 1000 == 0:
                    logger.info("Wrote %d tokenized lines", count)

    logger.info("Wrote %d tokenized lines to %s", count, output_path)
    return count


def train_model(
    *,
    corpus_path: Path,
    output_path: Path,
    analyzer: str,
    dim: int,
    epoch: int,
    min_count: int,
    minn: int,
    maxn: int,
    thread: int,
) -> None:
    try:
        import fasttext  # noqa: PLC0415
    except ImportError as exc:
        msg = "Install fasttext-wheel before training FastText models"
        raise RuntimeError(msg) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Training FastText model: %s", output_path)
    model = fasttext.train_unsupervised(
        str(corpus_path),
        model="skipgram",
        dim=dim,
        epoch=epoch,
        minCount=min_count,
        minn=minn,
        maxn=maxn,
        thread=thread,
    )
    model.save_model(str(output_path))
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "analyzer": analyzer,
        "corpus_path": str(corpus_path),
        "dim": dim,
        "epoch": epoch,
        "minCount": min_count,
        "minn": minn,
        "maxn": maxn,
    }
    output_path.with_suffix(output_path.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Saved %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build corpus and train FastText models for dedup benchmarks")
    parser.add_argument("--corpus-path", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--index", default=TEXT_INDEX_NAME)
    parser.add_argument("--analyzer", default=DEFAULT_ANALYZER)
    parser.add_argument("--limit", type=int, default=0, help="Limit OCR docs used for corpus; 0 means no limit")
    parser.add_argument("--skip-bocorpus", action="store_true")
    parser.add_argument("--reuse-corpus", action="store_true")
    parser.add_argument("--dim", type=int, default=100)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--thread", type=int, default=max(1, os.cpu_count() or 1))
    args = parser.parse_args()

    if not args.reuse_corpus or not args.corpus_path.exists():
        build_corpus(
            output_path=args.corpus_path,
            analyzer=args.analyzer,
            index_name=args.index,
            limit=args.limit,
            include_bocorpus=not args.skip_bocorpus,
        )

    train_model(
        corpus_path=args.corpus_path,
        output_path=args.output_dir / "bo_skipgram_subword.bin",
        analyzer=args.analyzer,
        dim=args.dim,
        epoch=args.epoch,
        min_count=args.min_count,
        minn=2,
        maxn=4,
        thread=args.thread,
    )
    train_model(
        corpus_path=args.corpus_path,
        output_path=args.output_dir / "bo_skipgram_nosubword.bin",
        analyzer=args.analyzer,
        dim=args.dim,
        epoch=args.epoch,
        min_count=args.min_count,
        minn=0,
        maxn=0,
        thread=args.thread,
    )


if __name__ == "__main__":
    main()

