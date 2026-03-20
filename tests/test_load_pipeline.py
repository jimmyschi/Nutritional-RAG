from __future__ import annotations

import tempfile
from pathlib import Path

from nutritional_rag.etl.models import ChunkedDocument, LoadPipelineConfig
from nutritional_rag.etl.pipeline import run_load_pipeline


def _write_chunks(path: Path, chunks: list[ChunkedDocument]) -> None:
    with path.open("w", encoding="utf-8") as file_handle:
        for chunk in chunks:
            file_handle.write(chunk.model_dump_json())
            file_handle.write("\n")


def _chunk(i: int) -> ChunkedDocument:
    return ChunkedDocument(
        chunk_id=f"chunk-{i}",
        document_id="doc-1",
        source_id="source-a",
        chunk_index=i,
        chunk_total=3,
        text=f"nutrition chunk text {i}",
        metadata={"nutrition_score": 3},
    )


def test_run_load_pipeline_dry_run_counts_all_chunks() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = Path(temp_dir) / "chunks.ndjson"
        _write_chunks(input_path, [_chunk(0), _chunk(1), _chunk(2)])

        summary = run_load_pipeline(
            LoadPipelineConfig(
                input_path=str(input_path),
                batch_size=2,
                dry_run=True,
            )
        )

        assert summary.total_chunks == 3
        assert summary.embedded_chunks == 3
        assert summary.upserted_vectors == 3
        assert summary.failed_chunks == 0
        assert summary.dry_run is True


def test_run_load_pipeline_empty_input() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = Path(temp_dir) / "chunks.ndjson"
        input_path.write_text("", encoding="utf-8")

        summary = run_load_pipeline(
            LoadPipelineConfig(
                input_path=str(input_path),
                dry_run=True,
            )
        )

        assert summary.total_chunks == 0
        assert summary.embedded_chunks == 0
        assert summary.upserted_vectors == 0
        assert summary.failed_chunks == 0
