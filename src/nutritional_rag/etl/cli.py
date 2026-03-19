from __future__ import annotations

import argparse
import json

from nutritional_rag.etl.models import (
    ChunkPipelineConfig,
    ExtractPipelineConfig,
    TransformPipelineConfig,
)
from nutritional_rag.etl.pipeline import (
    run_chunk_pipeline,
    run_extract_pipeline,
    run_transform_pipeline,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Nutritional RAG ETL stages")
    parser.add_argument("--stage", choices=["extract", "transform", "chunk"], default="extract")
    parser.add_argument(
        "--config",
        required=False,
        default="etl/sources.example.json",
        help="Path to extraction config JSON file",
    )
    parser.add_argument(
        "--input",
        required=False,
        default="data/raw/extracted_documents.ndjson",
        help="Input ndjson path for transform stage",
    )
    parser.add_argument(
        "--output",
        required=False,
        help="Optional output path override for the selected stage",
    )
    parser.add_argument(
        "--nutrition-only",
        action="store_true",
        help="When stage=transform, keep only nutrition-relevant documents",
    )
    parser.add_argument(
        "--min-nutrition-score",
        type=int,
        default=2,
        help="When stage=transform, minimum score threshold for nutrition filtering",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=400,
        help="When stage=chunk, target chunk size in tokens",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="When stage=chunk, overlap between consecutive chunks in tokens",
    )
    return parser.parse_args()


def _run_extract(config_path: str, output_override: str | None) -> None:
    with open(config_path, "r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)

    if output_override:
        payload["output_path"] = output_override

    config = ExtractPipelineConfig.model_validate(payload)
    summary = run_extract_pipeline(config)

    print("Extract pipeline finished")
    print(summary.model_dump_json(indent=2))


def _run_transform(
    input_path: str,
    output_override: str | None,
    nutrition_only: bool,
    min_nutrition_score: int,
) -> None:
    payload = {
        "input_path": input_path,
        "nutrition_only": nutrition_only,
        "min_nutrition_score": min_nutrition_score,
    }
    if output_override:
        payload["output_path"] = output_override

    config = TransformPipelineConfig.model_validate(payload)
    summary = run_transform_pipeline(config)

    print("Transform pipeline finished")
    print(summary.model_dump_json(indent=2))


def _run_chunk(
    input_path: str,
    output_override: str | None,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    payload: dict = {
        "input_path": input_path,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }
    if output_override:
        payload["output_path"] = output_override

    config = ChunkPipelineConfig.model_validate(payload)
    summary = run_chunk_pipeline(config)

    print("Chunk pipeline finished")
    print(summary.model_dump_json(indent=2))


def main() -> None:
    args = _parse_args()

    if args.stage == "extract":
        _run_extract(args.config, args.output)
        return

    if args.stage == "transform":
        _run_transform(args.input, args.output, args.nutrition_only, args.min_nutrition_score)
        return

    _run_chunk(args.input, args.output, args.chunk_size, args.chunk_overlap)


if __name__ == "__main__":
    main()
