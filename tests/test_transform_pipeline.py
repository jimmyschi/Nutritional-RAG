import json
from pathlib import Path

from nutritional_rag.etl.models import TransformPipelineConfig
from nutritional_rag.etl.pipeline import run_transform_pipeline


def test_transform_pipeline_reads_and_writes_ndjson(tmp_path: Path) -> None:
    input_path = tmp_path / "raw.ndjson"
    output_path = tmp_path / "processed.ndjson"

    record_a = {
        "document_id": "1",
        "source_id": "foods",
        "source_name": "foods",
        "source_location": "local",
        "text": "protein_g: 10\\nfat_g: 5",
    }
    record_b = {
        "document_id": "2",
        "source_id": "foods",
        "source_name": "foods",
        "source_location": "local",
        "text": "title: no nutrients",
    }

    input_path.write_text(
        "\n".join([json.dumps(record_a), json.dumps(record_b)]) + "\n",
        encoding="utf-8",
    )

    config = TransformPipelineConfig(input_path=str(input_path), output_path=str(output_path))
    summary = run_transform_pipeline(config)

    assert summary.total_documents == 2
    assert summary.transformed_documents == 2
    assert summary.documents_with_nutrients == 1
    assert output_path.exists()
    assert len(output_path.read_text(encoding="utf-8").splitlines()) == 2
