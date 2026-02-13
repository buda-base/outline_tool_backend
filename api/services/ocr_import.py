"""Import OCR results from parquet files stored in S3 into OpenSearch volumes."""

import logging
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import boto3
import pyarrow.parquet as pq

from api.config import Config
from api.models import Chunk, DocumentType, PageEntry, VolumeStatus
from api.services.opensearch import _index_document, _volume_doc_id

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1000


def _build_chunks(text: str, chunk_size: int = CHUNK_SIZE) -> list[Chunk]:
    """Split text into chunks of approximately chunk_size characters, breaking at newlines."""
    chunks: list[Chunk] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # Try to break at a newline boundary if not at the very end
        if end < text_len:
            newline_pos = text.rfind("\n", start, end)
            if newline_pos > start:
                end = newline_pos + 1

        chunks.append(
            Chunk(
                cstart=start,
                cend=end,
                text_bo=text[start:end],
            )
        )
        start = end

    return chunks


def _s3_key(w_id: str, i_id: str, i_version: str, source: str) -> str:
    """Build the S3 object key for an OCR parquet file."""
    filename = f"{w_id}-{i_id}-{i_version}_{source}.parquet"
    return f"{source}-ws-ldv1/{w_id}/{i_id}/{i_version}/{filename}"


def _download_from_s3(s3_key: str) -> Path:
    """Download a parquet file from S3 to a local temp file."""
    s3 = boto3.client("s3")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as tmp:
        tmp_path = tmp.name
    logger.info("Downloading s3://%s/%s", Config.S3_OCR_BUCKET, s3_key)
    s3.download_file(Config.S3_OCR_BUCKET, s3_key, tmp_path)
    return Path(tmp_path)


def import_ocr_from_s3(
    w_id: str,
    i_id: str,
    i_version: str,
    source: str,
) -> str:
    """
    Download a parquet OCR file from S3 and index the resulting volume into OpenSearch.

    Returns the document ID of the created volume.
    """
    key = _s3_key(w_id, i_id, i_version, source)
    local_path = _download_from_s3(key)
    try:
        return _import_parquet(w_id, i_id, i_version, source, local_path)
    finally:
        local_path.unlink(missing_ok=True)


def _import_parquet(
    w_id: str,
    i_id: str,
    i_version: str,
    source: str,
    parquet_path: Path,
) -> str:
    """
    Read a local parquet OCR file and index the resulting volume into OpenSearch.

    Returns the document ID of the created volume.
    """
    table = pq.read_table(parquet_path)
    logger.info("Read %d rows from %s", table.num_rows, parquet_path)

    # Collect successful pages: (filename, line_texts)
    pages_raw: list[tuple[str, list[str]]] = []
    skipped = 0
    for i in range(table.num_rows):
        ok = table.column("ok")[i].as_py()
        if not ok:
            skipped += 1
            continue
        fname = table.column("img_file_name")[i].as_py()
        lines = table.column("line_texts")[i].as_py() or []
        pages_raw.append((fname, lines))

    pages_raw.sort(key=lambda x: x[0])
    logger.info("Processing %d pages (%d skipped due to errors)", len(pages_raw), skipped)

    # Build continuous text and page entries
    full_text_parts: list[str] = []
    pages: list[PageEntry] = []
    offset = 0

    for pnum, (fname, lines) in enumerate(pages_raw, start=1):
        page_text = "\n".join(lines)
        cstart = offset
        cend = offset + len(page_text)

        pname = fname.rsplit(".", 1)[0]  # strip file extension
        pages.append(
            PageEntry(
                cstart=cstart,
                cend=cend,
                pnum=pnum,
                pname=pname,
            )
        )

        full_text_parts.append(page_text)
        offset = cend + 1  # +1 for the page separator newline

    full_text = "\n".join(full_text_parts)

    # Build search chunks
    chunks = _build_chunks(full_text)

    # Assemble and index the volume document
    now = datetime.now(UTC).isoformat()
    doc_id = _volume_doc_id(w_id, i_id)

    body = {
        "type": DocumentType.VOLUME_ETEXT.value,
        "w_id": w_id,
        "i_id": i_id,
        "i_version": i_version,
        "source": source,
        "status": VolumeStatus.NEW.value,
        "nb_pages": len(pages),
        "pages": [p.model_dump() for p in pages],
        "segments": [],
        "chunks": [c.model_dump() for c in chunks],
        "cstart": 0,
        "cend": len(full_text),
        "text": full_text,
        "first_imported_at": now,
        "last_updated_at": now,
        "join_field": {"name": "instance"},
    }

    _index_document(doc_id, body)
    logger.info(
        "Indexed volume %s: %d pages, %d chunks, %d characters",
        doc_id,
        len(pages),
        len(chunks),
        len(full_text),
    )
    return doc_id
