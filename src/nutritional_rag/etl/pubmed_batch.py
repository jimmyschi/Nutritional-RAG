from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from nutritional_rag.etl.extract import extract_source
from nutritional_rag.etl.models import (
    ChunkPipelineConfig,
    ExtractionSource,
    LoadPipelineConfig,
    RawDocument,
    TransformPipelineConfig,
)
from nutritional_rag.etl.pipeline import (
    run_chunk_pipeline,
    run_load_pipeline,
    run_transform_pipeline,
)


class PubMedTopic(BaseModel):
    query: str = Field(min_length=3)
    topic_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PubMedBatchConfig(BaseModel):
    topics: list[PubMedTopic]
    load_max_docs: int = 5
    sleep_seconds: float = 5.0
    failure_sleep_seconds: float = 20.0
    output_raw: str = "data/raw/pubmed_topics_extracted.ndjson"
    output_transformed: str = "data/processed/pubmed_topics_transformed.ndjson"
    output_chunks: str = "data/processed/pubmed_topics_chunks.ndjson"
    nutrition_only: bool = True
    min_nutrition_score: int = 2
    load_to_pinecone: bool = True
    batch_size: int = 50
    source_prefix: str = "pubmed-topic"
    source_name: str = "PubMed"
    source_license: str = "public-abstracts-check-terms"
    base_metadata: dict[str, Any] = Field(
        default_factory=lambda: {
            "domain": "general-nutrition",
            "source_type": "biomedical-literature",
        }
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_string_topics(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        topics = value.get("topics", [])
        coerced_topics = []
        for topic in topics:
            if isinstance(topic, str):
                coerced_topics.append({"query": topic})
            else:
                coerced_topics.append(topic)
        value["topics"] = coerced_topics
        return value


class PubMedBatchSummary(BaseModel):
    topic_count: int
    successful_topics: int
    failed_topics: list[str] = Field(default_factory=list)
    extracted_documents: int
    unique_documents: int
    output_raw: str
    output_transformed: str | None = None
    output_chunks: str | None = None
    loaded_to_pinecone: bool = False


def load_pubmed_batch_config(path: str | Path) -> PubMedBatchConfig:
    with Path(path).open("r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)
    return PubMedBatchConfig.model_validate(payload)


def run_pubmed_batch_pipeline(config: PubMedBatchConfig) -> PubMedBatchSummary:
    raw_documents: list[RawDocument] = []
    seen_keys: set[str] = set()
    failed_topics: list[str] = []
    successful_topics = 0

    for index, topic in enumerate(config.topics):
        if index > 0 and config.sleep_seconds > 0:
            time.sleep(config.sleep_seconds)

        source = _build_pubmed_source(config, topic)
        try:
            extracted_docs = extract_source(source)
            successful_topics += 1
        except Exception as error:
            failed_topics.append(f"{source.source_id}: {type(error).__name__}: {error}")
            if config.failure_sleep_seconds > 0:
                time.sleep(config.failure_sleep_seconds)
            continue

        for document in extracted_docs:
            dedupe_key = _document_dedupe_key(document)
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)
            raw_documents.append(document)

    _write_ndjson(Path(config.output_raw), raw_documents)

    output_transformed: str | None = None
    output_chunks: str | None = None
    loaded_to_pinecone = False

    if raw_documents:
        transform_summary = run_transform_pipeline(
            TransformPipelineConfig(
                input_path=config.output_raw,
                output_path=config.output_transformed,
                nutrition_only=config.nutrition_only,
                min_nutrition_score=config.min_nutrition_score,
            )
        )
        output_transformed = transform_summary.output_path

        chunk_summary = run_chunk_pipeline(
            ChunkPipelineConfig(
                input_path=config.output_transformed,
                output_path=config.output_chunks,
            )
        )
        output_chunks = chunk_summary.output_path

        if config.load_to_pinecone:
            run_load_pipeline(
                LoadPipelineConfig(
                    input_path=config.output_chunks,
                    batch_size=config.batch_size,
                    dry_run=False,
                )
            )
            loaded_to_pinecone = True

    return PubMedBatchSummary(
        topic_count=len(config.topics),
        successful_topics=successful_topics,
        failed_topics=failed_topics,
        extracted_documents=sum(1 for _ in raw_documents),
        unique_documents=len(raw_documents),
        output_raw=config.output_raw,
        output_transformed=output_transformed,
        output_chunks=output_chunks,
        loaded_to_pinecone=loaded_to_pinecone,
    )


def _build_pubmed_source(config: PubMedBatchConfig, topic: PubMedTopic) -> ExtractionSource:
    topic_id = topic.topic_id or f"{config.source_prefix}-{_slugify_query(topic.query)}"
    metadata = {
        **config.base_metadata,
        **topic.metadata,
        "load_max_docs": config.load_max_docs,
        "topic_query": topic.query,
    }
    return ExtractionSource(
        source_id=topic_id,
        kind="pubmed",
        location=topic.query,
        source_name=config.source_name,
        license=config.source_license,
        metadata=metadata,
    )


def _slugify_query(query: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", query.lower()).strip("-")
    return slug[:48] or "topic"


def _document_dedupe_key(document: RawDocument) -> str:
    for key in ("uid", "UID", "pmid", "PMID"):
        value = document.metadata.get(key)
        if value not in (None, ""):
            return f"pubmed:{value}"
    return document.document_id


def _write_ndjson(path: Path, documents: list[RawDocument]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_handle:
        for document in documents:
            file_handle.write(document.model_dump_json())
            file_handle.write("\n")