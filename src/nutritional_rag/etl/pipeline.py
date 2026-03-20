from __future__ import annotations

from pathlib import Path

from nutritional_rag.etl.chunk import chunk_document
from nutritional_rag.etl.extract import extract_source
from nutritional_rag.etl.load import (
    batch_iterable,
    chunk_to_metadata,
    deterministic_vector_id,
    embed_texts,
    get_openai_client,
    get_pinecone_index,
    iter_chunk_documents,
    resolve_load_config,
)
from nutritional_rag.etl.models import (
    ChunkPipelineConfig,
    ChunkRunSummary,
    ExtractPipelineConfig,
    ExtractRunSummary,
    LoadPipelineConfig,
    LoadRunSummary,
    RawDocument,
    TransformedDocument,
    TransformPipelineConfig,
    TransformRunSummary,
)
from nutritional_rag.etl.transform import transform_document


def run_extract_pipeline(config: ExtractPipelineConfig) -> ExtractRunSummary:
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_documents = []
    documents_by_source: dict[str, int] = {}

    for source in config.sources:
        extracted = extract_source(source)
        all_documents.extend(extracted)
        documents_by_source[source.source_id] = len(extracted)

    with output_path.open("w", encoding="utf-8") as file_handle:
        for document in all_documents:
            file_handle.write(document.model_dump_json())
            file_handle.write("\n")

    return ExtractRunSummary(
        output_path=str(output_path),
        total_documents=len(all_documents),
        documents_by_source=documents_by_source,
    )


def run_transform_pipeline(config: TransformPipelineConfig) -> TransformRunSummary:
    input_path = Path(config.input_path)
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    transformed_count = 0
    nutrient_count = 0
    total_documents = 0
    nutrition_candidate_count = 0
    filtered_out_count = 0

    with input_path.open("r", encoding="utf-8") as input_handle, output_path.open(
        "w", encoding="utf-8"
    ) as output_handle:
        for line in input_handle:
            payload = line.strip()
            if not payload:
                continue

            total_documents += 1
            raw_document = RawDocument.model_validate_json(payload)
            transformed = transform_document(raw_document)

            transformed_count += 1
            if transformed.nutrient_values:
                nutrient_count += 1

            is_nutrition_content = bool(transformed.metadata.get("is_nutrition_content", False))
            nutrition_score = int(transformed.metadata.get("nutrition_score", 0))

            if is_nutrition_content:
                nutrition_candidate_count += 1

            if config.nutrition_only and nutrition_score < config.min_nutrition_score:
                filtered_out_count += 1
                continue

            output_handle.write(transformed.model_dump_json())
            output_handle.write("\n")

    return TransformRunSummary(
        input_path=str(input_path),
        output_path=str(output_path),
        total_documents=total_documents,
        transformed_documents=transformed_count,
        documents_with_nutrients=nutrient_count,
        nutrition_candidate_documents=nutrition_candidate_count,
        filtered_out_documents=filtered_out_count,
    )


def run_chunk_pipeline(config: ChunkPipelineConfig) -> ChunkRunSummary:
    input_path = Path(config.input_path)
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_documents = 0
    total_chunks = 0

    with input_path.open("r", encoding="utf-8") as input_handle, output_path.open(
        "w", encoding="utf-8"
    ) as output_handle:
        for line in input_handle:
            payload = line.strip()
            if not payload:
                continue

            total_documents += 1
            doc = TransformedDocument.model_validate_json(payload)
            chunks = chunk_document(doc, config)

            for chunk in chunks:
                output_handle.write(chunk.model_dump_json())
                output_handle.write("\n")

            total_chunks += len(chunks)

    avg = round(total_chunks / total_documents, 2) if total_documents else 0.0

    return ChunkRunSummary(
        input_path=str(input_path),
        output_path=str(output_path),
        total_documents=total_documents,
        total_chunks=total_chunks,
        avg_chunks_per_document=avg,
    )


def run_load_pipeline(config: LoadPipelineConfig) -> LoadRunSummary:
    resolved_config = resolve_load_config(config)

    chunks = list(iter_chunk_documents(resolved_config.input_path))
    total_chunks = len(chunks)
    if total_chunks == 0:
        return LoadRunSummary(
            input_path=resolved_config.input_path,
            total_chunks=0,
            embedded_chunks=0,
            upserted_vectors=0,
            failed_chunks=0,
            dry_run=resolved_config.dry_run,
        )

    if not resolved_config.embedding_model:
        raise ValueError("Embedding model is required for load stage")

    if not resolved_config.dry_run and not resolved_config.pinecone_index:
        raise ValueError("Pinecone index is required for load stage unless --dry-run is set")

    embedded_chunks = 0
    upserted_vectors = 0
    failed_chunks = 0

    openai_client = None if resolved_config.dry_run else get_openai_client()
    index = None
    if not resolved_config.dry_run:
        index = get_pinecone_index(resolved_config.pinecone_index or "")

    for chunk_batch in batch_iterable(chunks, resolved_config.batch_size):
        texts = [chunk.text for chunk in chunk_batch]
        vector_ids = [deterministic_vector_id(chunk) for chunk in chunk_batch]

        if resolved_config.dry_run:
            embedded_chunks += len(chunk_batch)
            upserted_vectors += len(chunk_batch)
            continue

        try:
            embeddings = embed_texts(
                openai_client=openai_client,
                texts=texts,
                embedding_model=resolved_config.embedding_model,
            )

            vectors = []
            for chunk, vector_id, embedding in zip(
                chunk_batch, vector_ids, embeddings, strict=True
            ):
                vectors.append(
                    {
                        "id": vector_id,
                        "values": embedding,
                        "metadata": chunk_to_metadata(chunk),
                    }
                )

            index.upsert(vectors=vectors, namespace=resolved_config.pinecone_namespace)
            embedded_chunks += len(chunk_batch)
            upserted_vectors += len(chunk_batch)
        except Exception as exc:
            print(f"[load] batch failed: {type(exc).__name__}: {exc}")
            failed_chunks += len(chunk_batch)

    return LoadRunSummary(
        input_path=resolved_config.input_path,
        total_chunks=total_chunks,
        embedded_chunks=embedded_chunks,
        upserted_vectors=upserted_vectors,
        failed_chunks=failed_chunks,
        dry_run=resolved_config.dry_run,
    )
