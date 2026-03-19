from __future__ import annotations

import pytest

from nutritional_rag.etl.chunk import _chunk_words, chunk_document
from nutritional_rag.etl.models import ChunkPipelineConfig, TransformedDocument


def _make_doc(text: str, doc_id: str = "doc-1", source_id: str = "src-1") -> TransformedDocument:
    return TransformedDocument(
        document_id=doc_id,
        source_id=source_id,
        clean_text=text,
        metadata={"page_number": 1, "nutrition_score": 3},
    )


# ---------------------------------------------------------------------------
# _chunk_words unit tests
# ---------------------------------------------------------------------------


def test_chunk_words_single_chunk():
    words = "one two three".split()
    chunks = _chunk_words(words, chunk_size=400, overlap=50)
    assert len(chunks) == 1
    assert chunks[0] == words


def test_chunk_words_overlap_produces_repeated_words():
    # 10 words, chunk_size → ~7 words, overlap → ~3 words
    words = list("ABCDEFGHIJ")  # 10 single-char "words"
    chunks = _chunk_words(words, chunk_size=10, overlap=4)
    # Each successive chunk should share ~3 words with previous
    assert len(chunks) >= 2
    last_of_first = chunks[0][-3:]
    first_of_second = chunks[1][:3]
    assert last_of_first == first_of_second


def test_chunk_words_rejects_overlap_gte_size():
    with pytest.raises(ValueError, match="chunk_size"):
        _chunk_words(["a", "b"], chunk_size=5, overlap=5)


# ---------------------------------------------------------------------------
# chunk_document integration tests
# ---------------------------------------------------------------------------


def test_chunk_document_empty_text():
    doc = _make_doc("")
    config = ChunkPipelineConfig()
    assert chunk_document(doc, config) == []


def test_chunk_document_short_text_is_single_chunk():
    doc = _make_doc("This is a very short sentence.")
    config = ChunkPipelineConfig()
    chunks = chunk_document(doc, config)
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].chunk_total == 1
    assert chunks[0].document_id == "doc-1"
    assert chunks[0].source_id == "src-1"


def test_chunk_document_long_text_multiple_chunks():
    # ~600 words → should produce at least 2 chunks with default 400-token size
    words = " ".join([f"word{i}" for i in range(600)])
    doc = _make_doc(words)
    config = ChunkPipelineConfig(chunk_size=400, chunk_overlap=50)
    chunks = chunk_document(doc, config)
    assert len(chunks) >= 2
    # chunk_total should match actual count
    assert all(c.chunk_total == len(chunks) for c in chunks)
    # chunk_index should be sequential
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunk_document_metadata_propagated():
    doc = _make_doc("Protein intake is key for muscle recovery.")
    config = ChunkPipelineConfig()
    chunks = chunk_document(doc, config)
    for chunk in chunks:
        assert chunk.metadata["nutrition_score"] == 3
        assert chunk.metadata["page_number"] == 1
        assert "chunk_index" in chunk.metadata
        assert "chunk_total" in chunk.metadata
        assert "token_estimate" in chunk.metadata


def test_chunk_ids_are_unique():
    words = " ".join([f"w{i}" for i in range(600)])
    doc = _make_doc(words)
    config = ChunkPipelineConfig(chunk_size=200, chunk_overlap=20)
    chunks = chunk_document(doc, config)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "chunk_ids must be unique"
