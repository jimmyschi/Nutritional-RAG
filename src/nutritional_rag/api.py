from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from prometheus_client import make_asgi_app
from pydantic import BaseModel, Field

from nutritional_rag.settings import settings

app = FastAPI(title=settings.project_name, version="0.1.0")
app.mount("/metrics", make_asgi_app())


class QueryRequest(BaseModel):
    question: str = Field(min_length=3)
    top_k: int = Field(default=5, ge=1, le=20)


class Citation(BaseModel):
    vector_id: str
    score: float
    source_id: str | None = None
    document_id: str | None = None
    title: str | None = None
    page_number: int | None = None
    chunk_index: int | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]


def _require_rag_settings() -> None:
    if not settings.openai_api_key:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY is not configured")

    if not settings.pinecone_api_key:
        raise HTTPException(status_code=503, detail="PINECONE_API_KEY is not configured")

    if not settings.pinecone_index:
        raise HTTPException(status_code=503, detail="PINECONE_INDEX is not configured")


def _get_openai_client() -> Any:
    try:
        from openai import OpenAI
    except ModuleNotFoundError as error:
        raise HTTPException(status_code=500, detail="OpenAI package is not installed") from error

    return OpenAI(api_key=settings.openai_api_key)


def _get_pinecone_index() -> Any:
    try:
        from pinecone import Pinecone
    except ModuleNotFoundError as error:
        raise HTTPException(status_code=500, detail="Pinecone package is not installed") from error

    client = Pinecone(api_key=settings.pinecone_api_key)
    return client.Index(settings.pinecone_index)


def _embed_query(openai_client: Any, question: str) -> list[float]:
    response = openai_client.embeddings.create(
        model=settings.openai_embedding_model,
        input=[question],
    )
    return response.data[0].embedding


def _extract_matches(query_result: Any) -> list[Any]:
    if hasattr(query_result, "matches"):
        return list(query_result.matches or [])
    if isinstance(query_result, dict):
        return list(query_result.get("matches", []))
    return []


def _build_context_from_matches(matches: list[Any]) -> tuple[str, list[Citation]]:
    context_parts: list[str] = []
    citations: list[Citation] = []

    for idx, match in enumerate(matches, start=1):
        if isinstance(match, dict):
            metadata = match.get("metadata", {}) or {}
            vector_id = str(match.get("id", ""))
            score = float(match.get("score", 0.0))
        else:
            metadata = getattr(match, "metadata", {}) or {}
            vector_id = str(getattr(match, "id", ""))
            score = float(getattr(match, "score", 0.0))

        text = str(metadata.get("text", ""))
        if not text:
            continue

        context_parts.append(f"[{idx}] {text}")
        citations.append(
            Citation(
                vector_id=vector_id,
                score=score,
                source_id=metadata.get("source_id"),
                document_id=metadata.get("document_id"),
                title=metadata.get("title") or None,
                page_number=metadata.get("page_number"),
                chunk_index=metadata.get("chunk_index"),
            )
        )

    return "\n\n".join(context_parts), citations


def _generate_answer(openai_client: Any, question: str, context: str) -> str:
    system_prompt = (
        "You are a nutrition-focused RAG assistant. "
        "Answer using only the provided context. "
        "If the context does not contain enough information, say so clearly. "
        "Do not fabricate studies or citations."
    )

    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Context snippets:\n{context}\n\n"
        "Provide a concise, practical answer and mention uncertainty when needed."
    )

    response = openai_client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content or "No answer generated."


@app.get("/healthz")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
def readiness() -> dict[str, bool | str]:
    return {
        "status": "ready",
        "openai_configured": bool(settings.openai_api_key),
        "pinecone_configured": bool(settings.pinecone_api_key and settings.pinecone_index),
        "redis_configured": bool(settings.redis_url),
        "mlflow_configured": bool(settings.mlflow_tracking_uri),
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    _require_rag_settings()

    try:
        openai_client = _get_openai_client()
        pinecone_index = _get_pinecone_index()

        query_vector = _embed_query(openai_client, request.question)
        query_result = pinecone_index.query(
            vector=query_vector,
            top_k=request.top_k,
            include_metadata=True,
            namespace=settings.pinecone_namespace,
        )

        matches = _extract_matches(query_result)
        context, citations = _build_context_from_matches(matches)

        if not context:
            return QueryResponse(
                answer="I could not find relevant context in the current knowledge base.",
                citations=[],
            )

        answer = _generate_answer(openai_client, request.question, context)
        return QueryResponse(answer=answer, citations=citations)
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Query failed: {error}") from error
