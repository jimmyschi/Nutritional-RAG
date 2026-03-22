from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


class ExtractionSource(BaseModel):
    source_id: str
    kind: Literal["csv", "json", "html", "text", "pdf", "pubmed", "youtube", "web"]
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


class TransformedDocument(BaseModel):
    document_id: str
    source_id: str
    title: str | None = None
    clean_text: str
    nutrient_values: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TransformPipelineConfig(BaseModel):
    input_path: str = "data/raw/extracted_documents.ndjson"
    output_path: str = "data/processed/transformed_documents.ndjson"
    nutrition_only: bool = False
    min_nutrition_score: int = 2


class TransformRunSummary(BaseModel):
    input_path: str
    output_path: str
    total_documents: int
    transformed_documents: int
    documents_with_nutrients: int
    nutrition_candidate_documents: int
    filtered_out_documents: int


class ChunkedDocument(BaseModel):
    chunk_id: str
    document_id: str
    source_id: str
    title: str | None = None
    chunk_index: int
    chunk_total: int
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkPipelineConfig(BaseModel):
    input_path: str = "data/processed/transformed_documents.ndjson"
    output_path: str = "data/processed/chunks.ndjson"
    chunk_size: int = 400
    chunk_overlap: int = 50


class ChunkRunSummary(BaseModel):
    input_path: str
    output_path: str
    total_documents: int
    total_chunks: int
    avg_chunks_per_document: float


class LoadPipelineConfig(BaseModel):
    input_path: str = "data/processed/chunks.ndjson"
    embedding_model: str | None = None
    pinecone_index: str | None = None
    pinecone_namespace: str | None = None
    batch_size: int = 100
    dry_run: bool = False


class LoadRunSummary(BaseModel):
    input_path: str
    total_chunks: int
    embedded_chunks: int
    upserted_vectors: int
    failed_chunks: int
    dry_run: bool
