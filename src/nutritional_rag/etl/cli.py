from __future__ import annotations

import argparse
import json

from nutritional_rag.etl.models import ExtractPipelineConfig
from nutritional_rag.etl.pipeline import run_extract_pipeline


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Nutritional RAG extract pipeline")
    parser.add_argument("--config", required=True, help="Path to ETL config JSON file")
    parser.add_argument(
        "--output",
        required=False,
        help="Optional output path override for extracted ndjson documents",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    with open(args.config, "r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)

    if args.output:
        payload["output_path"] = args.output

    config = ExtractPipelineConfig.model_validate(payload)
    summary = run_extract_pipeline(config)

    print("Extract pipeline finished")
    print(summary.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
