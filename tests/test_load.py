from __future__ import annotations

from nutritional_rag.etl.load import batch_iterable, chunk_to_metadata, deterministic_vector_id
from nutritional_rag.etl.models import ChunkedDocument


def _make_chunk(chunk_index: int = 0) -> ChunkedDocument:
    return ChunkedDocument(
        chunk_id="chunk-random",
        document_id="doc-1",
        source_id="source-a",
        chunk_index=chunk_index,
        chunk_total=3,
        text="protein and carbs",
        metadata={"page_number": 1},
    )


def test_deterministic_vector_id_stable_for_same_chunk():
    chunk = _make_chunk(chunk_index=1)
    assert deterministic_vector_id(chunk) == deterministic_vector_id(chunk)


def test_deterministic_vector_id_changes_by_chunk_index():
    first = _make_chunk(chunk_index=0)
    second = _make_chunk(chunk_index=1)
    assert deterministic_vector_id(first) != deterministic_vector_id(second)


def test_batch_iterable_groups_items():
    values = [1, 2, 3, 4, 5]
    batches = list(batch_iterable(values, 2))
    assert batches == [[1, 2], [3, 4], [5]]


def test_chunk_to_metadata_sanitizes_nested_values_for_pinecone():
    chunk = ChunkedDocument(
        chunk_id="chunk-random",
        document_id="doc-1",
        source_id="pubmed-sports-nutrition",
        chunk_index=0,
        chunk_total=1,
        text="Creatine may improve high-intensity performance.",
        metadata={
            "Title": {"#text": "Creatine and Performance"},
            "authors": ["Smith", 42],
            "pmid": 12345,
        },
    )

    metadata = chunk_to_metadata(chunk)

    assert metadata["Title"] == '{"#text": "Creatine and Performance"}'
    assert metadata["authors"] == ["Smith", "42"]
    assert metadata["pmid"] == 12345
