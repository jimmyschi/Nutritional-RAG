from pathlib import Path

from nutritional_rag.etl.models import ExtractPipelineConfig
from nutritional_rag.etl.pipeline import run_extract_pipeline


def test_extract_pipeline_writes_ndjson(tmp_path: Path) -> None:
    output_path = tmp_path / "documents.ndjson"
    config = ExtractPipelineConfig.model_validate(
        {
            "output_path": str(output_path),
            "sources": [
                {
                    "source_id": "foods",
                    "kind": "csv",
                    "location": "tests/fixtures/extract/foods.csv",
                    "source_name": "foods",
                },
                {
                    "source_id": "guidelines",
                    "kind": "json",
                    "location": "tests/fixtures/extract/guidelines.json",
                    "source_name": "guidelines",
                },
            ],
        }
    )

    summary = run_extract_pipeline(config)

    assert summary.total_documents == 5
    assert output_path.exists()
    assert len(output_path.read_text(encoding="utf-8").splitlines()) == 5
