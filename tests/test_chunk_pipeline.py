from __future__ import annotations

import json
import tempfile
from pathlib import Path

from nutritional_rag.etl.models import ChunkPipelineConfig, TransformedDocument
from nutritional_rag.etl.pipeline import run_chunk_pipeline


def _write_transformed_docs(path: Path, docs: list[TransformedDocument]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for doc in docs:
            fh.write(doc.model_dump_json())
            fh.write("\n")


def test_run_chunk_pipeline_produces_chunks():
    words = " ".join([f"word{i}" for i in range(600)])
    docs = [
        TransformedDocument(
            document_id="doc-1",
            source_id="src-1",
            clean_text=words,
            metadata={"page_number": 1, "nutrition_score": 3},
        ),
        TransformedDocument(
            document_id="doc-2",
            source_id="src-1",
            clean_text="Short text.",
            metadata={"page_number": 2, "nutrition_score": 1},
        ),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "transformed.ndjson"
        output_path = Path(tmpdir) / "chunks.ndjson"
        _write_transformed_docs(input_path, docs)

        config = ChunkPipelineConfig(
            input_path=str(input_path),
            output_path=str(output_path),
            chunk_size=400,
            chunk_overlap=50,
        )
        summary = run_chunk_pipeline(config)

        assert summary.total_documents == 2
        assert summary.total_chunks >= 3  # doc-1 → ≥2 chunks, doc-2 → 1 chunk
        assert summary.avg_chunks_per_document > 0

        lines = output_path.read_text().strip().splitlines()
        assert len(lines) == summary.total_chunks

        first_chunk = json.loads(lines[0])
        assert "chunk_id" in first_chunk
        assert "chunk_index" in first_chunk
        assert "text" in first_chunk


def test_run_chunk_pipeline_empty_input():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "empty.ndjson"
        output_path = Path(tmpdir) / "chunks.ndjson"
        input_path.write_text("")

        config = ChunkPipelineConfig(
            input_path=str(input_path),
            output_path=str(output_path),
        )
        summary = run_chunk_pipeline(config)

        assert summary.total_documents == 0
        assert summary.total_chunks == 0
        assert summary.avg_chunks_per_document == 0.0
