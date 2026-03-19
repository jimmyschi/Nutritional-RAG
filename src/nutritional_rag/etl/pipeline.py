from __future__ import annotations

from pathlib import Path

from nutritional_rag.etl.extract import extract_source
from nutritional_rag.etl.models import (
    ExtractPipelineConfig,
    ExtractRunSummary,
    RawDocument,
    TransformPipelineConfig,
    TransformRunSummary,
)
from nutritional_rag.etl.transform import transform_document


def run_extract_pipeline(config: ExtractPipelineConfig) -> ExtractRunSummary:
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_documents = []
    documents_by_source: dict[str, int] = {}

    for source in config.sources:
        extracted = extract_source(source)
        all_documents.extend(extracted)
        documents_by_source[source.source_id] = len(extracted)

    with output_path.open("w", encoding="utf-8") as file_handle:
        for document in all_documents:
            file_handle.write(document.model_dump_json())
            file_handle.write("\n")

    return ExtractRunSummary(
        output_path=str(output_path),
        total_documents=len(all_documents),
        documents_by_source=documents_by_source,
    )


def run_transform_pipeline(config: TransformPipelineConfig) -> TransformRunSummary:
    input_path = Path(config.input_path)
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    transformed_count = 0
    nutrient_count = 0
    total_documents = 0

    with input_path.open("r", encoding="utf-8") as input_handle, output_path.open(
        "w", encoding="utf-8"
    ) as output_handle:
        for line in input_handle:
            payload = line.strip()
            if not payload:
                continue

            total_documents += 1
            raw_document = RawDocument.model_validate_json(payload)
            transformed = transform_document(raw_document)

            transformed_count += 1
            if transformed.nutrient_values:
                nutrient_count += 1

            output_handle.write(transformed.model_dump_json())
            output_handle.write("\n")

    return TransformRunSummary(
        input_path=str(input_path),
        output_path=str(output_path),
        total_documents=total_documents,
        transformed_documents=transformed_count,
        documents_with_nutrients=nutrient_count,
    )
