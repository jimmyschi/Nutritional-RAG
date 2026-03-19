from __future__ import annotations

import re
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from nutritional_rag.etl.models import ChunkedDocument, ChunkPipelineConfig, TransformedDocument

# Approximate word-to-token ratio for English prose.
# tiktoken would be more accurate, but avoids an extra dependency.
_WORDS_PER_TOKEN = 0.75


def _word_tokenize(text: str) -> list[str]:
    """Split text into whitespace-delimited tokens (words)."""
    return text.split()


def _words_to_token_estimate(word_count: int) -> int:
    return int(word_count / _WORDS_PER_TOKEN)


def _chunk_words(words: list[str], chunk_size: int, overlap: int) -> list[list[str]]:
    """
    Slide a window over *words* with the given chunk_size (in tokens) and overlap.

    chunk_size and overlap are expressed in *token equivalents*; we convert them to
    approximate word counts before slicing.
    """
    word_chunk = max(1, int(chunk_size * _WORDS_PER_TOKEN))
    word_overlap = max(0, int(overlap * _WORDS_PER_TOKEN))

    if word_chunk <= word_overlap:
        raise ValueError(f"chunk_size ({chunk_size}) must be larger than chunk_overlap ({overlap})")

    chunks: list[list[str]] = []
    start = 0
    while start < len(words):
        end = start + word_chunk
        chunks.append(words[start:end])
        if end >= len(words):
            break
        start += word_chunk - word_overlap

    return chunks


def _clean_chunk_text(text: str) -> str:
    """Collapse internal whitespace and strip outer whitespace."""
    return re.sub(r"\s+", " ", text).strip()


def chunk_document(
    doc: TransformedDocument,
    config: ChunkPipelineConfig,
) -> list[ChunkedDocument]:
    """Split a single TransformedDocument into ChunkedDocuments."""
    words = _word_tokenize(doc.clean_text)

    if not words:
        return []

    word_groups = _chunk_words(words, config.chunk_size, config.chunk_overlap)
    total = len(word_groups)

    chunks: list[ChunkedDocument] = []
    for idx, word_group in enumerate(word_groups):
        chunk_text = _clean_chunk_text(" ".join(word_group))
        chunk_id = str(uuid.uuid4())

        chunk_meta: dict = {
            **doc.metadata,
            "chunk_index": idx,
            "chunk_total": total,
            "token_estimate": _words_to_token_estimate(len(word_group)),
        }

        chunks.append(
            ChunkedDocument(
                chunk_id=chunk_id,
                document_id=doc.document_id,
                source_id=doc.source_id,
                title=doc.title,
                chunk_index=idx,
                chunk_total=total,
                text=chunk_text,
                metadata=chunk_meta,
            )
        )

    return chunks
