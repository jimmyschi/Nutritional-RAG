from __future__ import annotations

import argparse

from nutritional_rag.etl.pubmed_batch import load_pubmed_batch_config, run_pubmed_batch_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch PubMed ETL across multiple nutrition topics with cooldowns"
    )
    parser.add_argument(
        "--config",
        default="etl/pubmed_topics.example.json",
        help="Path to topic batch config JSON file",
    )
    args = parser.parse_args()

    config = load_pubmed_batch_config(args.config)
    summary = run_pubmed_batch_pipeline(config)
    print(summary.model_dump_json(indent=2))


if __name__ == "__main__":
    main()