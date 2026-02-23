from __future__ import annotations

import secrets
import string
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class VolumeStatus(StrEnum):
    NEW = "new"
    IN_PROGRESS = "in_progress"
    REVIEW = "review"
    COMPLETED = "completed"


class SegmentType(StrEnum):
    TEXT = "text"
    MAIN_TEXT = "main_text"
    EDITORIAL = "editorial"


class DocumentType(StrEnum):
    VOLUME_ETEXT = "volume_etext"
    WORK = "work"
    PERSON = "person"


class RecordStatus(StrEnum):
    ACTIVE = "active"
    DUPLICATE = "duplicate"
    WITHDRAWN = "withdrawn"


class Origin(StrEnum):
    IMPORTED = "imported"
    LOCAL = "local"


_ID_ALPHABET = string.ascii_uppercase + string.digits


def generate_id(prefix: str, length: int = 7) -> str:
    suffix = "".join(secrets.choice(_ID_ALPHABET) for _ in range(length))
    return f"{prefix}{suffix}"


class ParsedRecord(BaseModel):
    id: str
    type: str
    is_released: bool
    replacement_id: str | None = None
    pref_label_bo: str | None = None
    alt_label_bo: list[str] = Field(default_factory=list)
    author: str | None = None


class ImportRecord(BaseModel):
    id: str
    type: str
    pref_label_bo: str | None = None
    alt_label_bo: list[str] = Field(default_factory=list)
    author: str | None = None
    db_score: float | None = None


class SyncCounts(BaseModel):
    upserted: int = 0
    merged: int = 0
    withdrawn: int = 0
    skipped: int = 0


class Chunk(BaseModel):
    cstart: int
    cend: int
    text_bo: str | None = None


class PageEntry(BaseModel):
    cstart: int
    cend: int
    pnum: int | None = None
    pname: str | None = None
    img_link: str | None = None


class Segment(BaseModel):
    id: str | None = None
    cstart: int
    cend: int
    segment_type: SegmentType = SegmentType.TEXT
    parent_segment: str | None = None
    title_bo: str | None = None
    author_name_bo: str | None = None


class CurationMeta(BaseModel):
    modified: bool = False
    modified_at: datetime | None = None
    modified_by: str | None = None
    edit_version: int = 0


class SourceMeta(BaseModel):
    updated_at: datetime | None = None


class ImportMeta(BaseModel):
    last_run_at: datetime | None = None
    last_result: str | None = None


class RecordOutput(BaseModel):
    id: str
    origin: Origin | None = None
    record_status: RecordStatus | None = None
    canonical_id: str | None = None
    curation: CurationMeta | None = None
    source_meta: SourceMeta | None = None
    import_meta: ImportMeta | None = None
    db_score: float | None = None


class PersonBase(BaseModel):
    pref_label_bo: str | None = None
    alt_label_bo: str | None = None
    dates: str | None = None


class PersonInput(PersonBase):
    modified_by: str


class PersonOutput(PersonBase, RecordOutput):
    pass


class WorkBase(BaseModel):
    pref_label_bo: str | None = None
    alt_label_bo: str | None = None
    author: str | None = None
    versions: list[str] = Field(default_factory=list)


class WorkInput(WorkBase):
    modified_by: str


class WorkOutput(WorkBase, RecordOutput):
    pass


class MergeRequest(BaseModel):
    canonical_id: str
    modified_by: str


class VolumeBase(BaseModel):
    vol_version: str | None = None
    etext_source: str | None = None
    volume_number: int | None = None
    wa_id: str | None = None
    mw_id: str | None = None
    status: VolumeStatus = VolumeStatus.NEW
    pages: list[PageEntry] = Field(default_factory=list)
    segments: list[Segment] = Field(default_factory=list)


class VolumeInput(VolumeBase):
    pass


class VolumeOutput(VolumeBase):
    id: str
    rep_id: str
    vol_id: str
    nb_pages: int | None = None
    first_imported_at: datetime | None = None
    last_updated_at: datetime | None = None
    replaced_by: str | None = None
    cstart: int | None = None
    cend: int | None = None
    chunks: list[Chunk] = Field(default_factory=list)


class PaginatedResponse(BaseModel):
    total: int
    offset: int
    limit: int
    items: list[VolumeOutput] = Field(default_factory=list)


class ImportOCRRequest(BaseModel):
    rep_id: str
    vol_id: str
    vol_version: str
    etext_source: str


class CatalogBreakdown(BaseModel):
    with_preexisting_catalog: int = 0
    no_preexisting_catalog: int = 0


class Stats(BaseModel):
    nb_volumes_imported: CatalogBreakdown = Field(default_factory=CatalogBreakdown)
    nb_volumes_finished: CatalogBreakdown = Field(default_factory=CatalogBreakdown)
    nb_segments_total: int = 0
    nb_works_total: int = 0
    nb_persons_total: int = 0
    nb_merges_identified: int = 0
