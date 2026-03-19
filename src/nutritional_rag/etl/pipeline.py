from __future__ import annotations

from pathlib import Path

from nutritional_rag.etl.extract import extract_source
from nutritional_rag.etl.models import ExtractPipelineConfig, ExtractRunSummary


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
