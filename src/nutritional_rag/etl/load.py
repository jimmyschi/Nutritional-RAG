from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, TypeVar

from nutritional_rag.etl.models import ChunkedDocument, LoadPipelineConfig
from nutritional_rag.settings import settings


def deterministic_vector_id(chunk: ChunkedDocument) -> str:
    """Create a stable vector id for upserts/dedup across reruns."""
    base = f"{chunk.source_id}:{chunk.document_id}:{chunk.chunk_index}"
    digest = hashlib.sha256(base.encode("utf-8", errors="ignore")).hexdigest()[:24]
    return f"{chunk.source_id}:{chunk.document_id}:{chunk.chunk_index}:{digest}"


def resolve_load_config(config: LoadPipelineConfig) -> LoadPipelineConfig:
    embedding_model = config.embedding_model or settings.openai_embedding_model
    pinecone_index = config.pinecone_index or settings.pinecone_index
    pinecone_namespace = config.pinecone_namespace or settings.pinecone_namespace

    return LoadPipelineConfig(
        input_path=config.input_path,
        embedding_model=embedding_model,
        pinecone_index=pinecone_index,
        pinecone_namespace=pinecone_namespace,
        batch_size=config.batch_size,
        dry_run=config.dry_run,
    )


def iter_chunk_documents(path: str) -> Iterator[ChunkedDocument]:
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as file_handle:
        for line in file_handle:
            payload = line.strip()
            if not payload:
                continue
            yield ChunkedDocument.model_validate_json(payload)


T = TypeVar("T")


def batch_iterable(items: Iterable[T], batch_size: int) -> Iterator[list[T]]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")

    batch: list[T] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


def chunk_to_metadata(chunk: ChunkedDocument) -> dict[str, Any]:
    metadata = {
        "source_id": chunk.source_id,
        "document_id": chunk.document_id,
        "chunk_id": chunk.chunk_id,
        "chunk_index": chunk.chunk_index,
        "chunk_total": chunk.chunk_total,
        "title": chunk.title or "",
    }
    metadata.update(chunk.metadata)
    return metadata


def get_openai_client() -> Any:
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is required for load stage unless --dry-run is set")

    try:
        from openai import OpenAI
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Load stage requires 'openai'. Install with: pip install openai"
        ) from error

    return OpenAI(api_key=settings.openai_api_key)


def get_pinecone_index(index_name: str) -> Any:
    if not settings.pinecone_api_key:
        raise ValueError("PINECONE_API_KEY is required for load stage unless --dry-run is set")

    try:
        from pinecone import Pinecone
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Load stage requires 'pinecone'. Install with: pip install pinecone"
        ) from error

    client = Pinecone(api_key=settings.pinecone_api_key)
    return client.Index(index_name)


def embed_texts(
    openai_client: Any,
    texts: Sequence[str],
    embedding_model: str,
) -> list[list[float]]:
    response = openai_client.embeddings.create(model=embedding_model, input=list(texts))
    return [item.embedding for item in response.data]
