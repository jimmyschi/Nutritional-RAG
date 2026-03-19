from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


class ExtractionSource(BaseModel):
    source_id: str
    kind: Literal["csv", "json", "html", "text"]
    location: str
    source_name: str
    license: str | None = None
    language: str = "en"
    metadata: dict[str, Any] = Field(default_factory=dict)


class RawDocument(BaseModel):
    document_id: str
    source_id: str
    source_name: str
    source_location: str
    extracted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    title: str | None = None
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExtractPipelineConfig(BaseModel):
    sources: list[ExtractionSource]
    output_path: str = "data/raw/extracted_documents.ndjson"


class ExtractRunSummary(BaseModel):
    output_path: str
    total_documents: int
    documents_by_source: dict[str, int]
