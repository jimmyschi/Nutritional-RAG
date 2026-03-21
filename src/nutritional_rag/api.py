from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Response
from prometheus_client import REGISTRY, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

from nutritional_rag.settings import settings

app = FastAPI(title=settings.project_name, version="0.1.0")


QUERY_REQUESTS_TOTAL = Counter(
    "nutritional_rag_query_requests_total",
    "Total number of query requests by outcome.",
    ["status", "cache_hit", "generate_answer"],
)
QUERY_ERRORS_TOTAL = Counter(
    "nutritional_rag_query_errors_total",
    "Total number of query errors by error type.",
    ["error_type"],
)
QUERY_CACHE_CHECKS_TOTAL = Counter(
    "nutritional_rag_query_cache_checks_total",
    "Cache checks by result.",
    ["result"],
)
QUERY_IN_FLIGHT = Gauge(
    "nutritional_rag_query_in_flight",
    "Current number of in-flight query requests.",
)
QUERY_DURATION_SECONDS = Histogram(
    "nutritional_rag_query_duration_seconds",
    "End-to-end query latency in seconds.",
    buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60, 120),
)
QUERY_TOP_K = Histogram(
    "nutritional_rag_query_top_k",
    "Requested top_k values.",
    buckets=(1, 3, 5, 8, 10, 12, 15, 20),
)
QUERY_RERANK_MULTIPLIER = Histogram(
    "nutritional_rag_query_rerank_candidate_multiplier",
    "Effective rerank candidate multiplier values.",
    buckets=(1, 2, 3, 5, 8, 10, 15, 20),
)
QUERY_CANDIDATE_MATCHES = Histogram(
    "nutritional_rag_query_candidate_matches",
    "Number of candidate matches returned from Pinecone before reranking.",
    buckets=(0, 1, 3, 5, 8, 10, 15, 20, 30, 50, 100),
)
QUERY_CITATIONS_RETURNED = Histogram(
    "nutritional_rag_query_citations_returned",
    "Number of citations returned to clients.",
    buckets=(0, 1, 3, 5, 8, 10, 15, 20),
)
QUERY_MEAN_CITATION_SCORE = Histogram(
    "nutritional_rag_query_mean_citation_score",
    "Mean citation score per response.",
    buckets=(0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)


SOURCE_TITLE_OVERRIDES: dict[str, str] = {
    "exercise-physiology-book-pdf": (
        "Exercise Physiology: Human Bioenergetics and Its Applications "
        "(Brooks, Fahey, Baldwin; 4th Edition, 2004)"
    )
}


class QueryRequest(BaseModel):
    question: str = Field(min_length=3)
    top_k: int = Field(default=5, ge=1, le=20)
    rerank_candidate_multiplier: int | None = Field(default=None, ge=1, le=20)
    use_cache: bool = True
    generate_answer: bool = True


class Citation(BaseModel):
    vector_id: str
    score: float
    source_id: str | None = None
    document_id: str | None = None
    title: str | None = None
    page_number: int | None = None
    chunk_index: int | None = None
    pubmed_url: str | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    cache_hit: bool = False


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


def _get_redis_client() -> Any | None:
    try:
        import redis
    except ModuleNotFoundError:
        return None

    try:
        return redis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=1,
            socket_timeout=1,
        )
    except Exception:
        return None


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


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _match_text(match: Any) -> str:
    if isinstance(match, dict):
        metadata = match.get("metadata", {}) or {}
    else:
        metadata = getattr(match, "metadata", {}) or {}
    return str(metadata.get("text", ""))


def _match_score(match: Any) -> float:
    if isinstance(match, dict):
        return float(match.get("score", 0.0))
    return float(getattr(match, "score", 0.0))


def _rerank_matches(question: str, matches: list[Any], top_k: int) -> list[Any]:
    question_tokens = _tokenize(question)
    if not question_tokens:
        return matches[:top_k]

    ranked: list[tuple[float, Any]] = []
    for match in matches:
        text_tokens = _tokenize(_match_text(match))
        overlap = len(question_tokens & text_tokens)
        lexical = overlap / max(1, len(question_tokens))
        pinecone_sim = _match_score(match)
        blended = 0.8 * pinecone_sim + 0.2 * lexical
        ranked.append((blended, match))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in ranked[:top_k]]


def _effective_rerank_candidate_multiplier(request: QueryRequest) -> int:
    return request.rerank_candidate_multiplier or settings.rerank_candidate_multiplier


def _cache_key(request: QueryRequest) -> str:
    payload = (
        f"{settings.pinecone_namespace}:{request.top_k}:"
        f"{_effective_rerank_candidate_multiplier(request)}:"
        f"{request.question.strip().lower()}"
    )
    digest = hashlib.sha256(payload.encode("utf-8", errors="ignore")).hexdigest()[:24]
    return f"query:{digest}"


def _read_query_cache(redis_client: Any, cache_key: str) -> QueryResponse | None:
    if redis_client is None:
        return None

    try:
        cached = redis_client.get(cache_key)
    except Exception:
        return None

    if not cached:
        return None

    try:
        payload = json.loads(cached)
        payload["cache_hit"] = True
        return QueryResponse.model_validate(payload)
    except Exception:
        return None


def _write_query_cache(redis_client: Any, cache_key: str, response: QueryResponse) -> None:
    if redis_client is None:
        return

    payload = response.model_dump()
    payload["cache_hit"] = False

    try:
        redis_client.setex(cache_key, settings.query_cache_ttl_seconds, json.dumps(payload))
    except Exception:
        return


def _get_mlflow_client() -> Any | None:
    if not settings.mlflow_log_queries:
        return None

    try:
        import mlflow
    except ModuleNotFoundError:
        return None

    return mlflow


def _log_query_to_mlflow(
    request: QueryRequest,
    response: QueryResponse | None,
    *,
    latency_ms: float,
    candidate_count: int,
    status: str,
    error_detail: str | None = None,
) -> None:
    mlflow = _get_mlflow_client()
    if mlflow is None or not settings.mlflow_tracking_uri:
        return

    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)

        with mlflow.start_run(run_name="query"):
            mlflow.log_params(
                {
                    "top_k": request.top_k,
                    "question_length_chars": len(request.question),
                    "openai_model": settings.openai_model,
                    "embedding_model": settings.openai_embedding_model,
                    "namespace": settings.pinecone_namespace,
                    "rerank_candidate_multiplier": _effective_rerank_candidate_multiplier(request),
                    "use_cache": request.use_cache,
                    "generate_answer": request.generate_answer,
                    "cache_ttl_seconds": settings.query_cache_ttl_seconds,
                }
            )
            mlflow.log_metrics(
                {
                    "latency_ms": latency_ms,
                    "candidate_count": float(candidate_count),
                    "status_ok": 1.0 if status == "ok" else 0.0,
                }
            )
            mlflow.set_tags(
                {
                    "status": status,
                    "app_env": settings.app_env,
                }
            )

            if response is not None:
                scores = [citation.score for citation in response.citations]
                mlflow.log_metrics(
                    {
                        "cache_hit": 1.0 if response.cache_hit else 0.0,
                        "citation_count": float(len(response.citations)),
                        "mean_citation_score": (sum(scores) / len(scores)) if scores else 0.0,
                        "max_citation_score": max(scores) if scores else 0.0,
                        "min_citation_score": min(scores) if scores else 0.0,
                    }
                )

            if error_detail:
                mlflow.set_tag("error_detail", error_detail[:500])
    except Exception:
        return


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
        source_id = metadata.get("source_id")
        resolved_title = metadata.get("title") or SOURCE_TITLE_OVERRIDES.get(str(source_id or ""))
        pubmed_url = _pubmed_url_from_metadata(metadata)

        citations.append(
            Citation(
                vector_id=vector_id,
                score=score,
                source_id=source_id,
                document_id=metadata.get("document_id"),
                title=resolved_title or None,
                page_number=metadata.get("page_number"),
                chunk_index=metadata.get("chunk_index"),
                pubmed_url=pubmed_url,
            )
        )

    return "\n\n".join(context_parts), citations


def _pubmed_url_from_metadata(metadata: dict[str, Any]) -> str | None:
    source_id = str(metadata.get("source_id", ""))
    if "pubmed" not in source_id.lower():
        return None

    for key in ("uid", "UID", "pmid", "PMID"):
        value = metadata.get(key)
        if value in (None, ""):
            continue
        identifier = str(value).strip()
        if identifier:
            return f"https://pubmed.ncbi.nlm.nih.gov/{identifier}/"

    return None


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


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(REGISTRY), media_type="text/plain; version=0.0.4")


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
    started_at = time.perf_counter()
    candidate_count = 0
    effective_multiplier = max(1, _effective_rerank_candidate_multiplier(request))
    cache_result = "skipped"
    response_obj: QueryResponse | None = None
    error_type: str | None = None
    status = "ok"
    QUERY_IN_FLIGHT.inc()

    try:
        redis_client = _get_redis_client()
        key = _cache_key(request)
        if request.use_cache:
            cache_result = "miss"
            cached = _read_query_cache(redis_client, key)
            if cached is not None:
                cache_result = "hit"
                response_obj = cached
                _log_query_to_mlflow(
                    request,
                    cached,
                    latency_ms=(time.perf_counter() - started_at) * 1000,
                    candidate_count=0,
                    status="ok",
                )
                return cached

        openai_client = _get_openai_client()
        pinecone_index = _get_pinecone_index()

        query_vector = _embed_query(openai_client, request.question)
        candidate_top_k = request.top_k * effective_multiplier
        query_result = pinecone_index.query(
            vector=query_vector,
            top_k=candidate_top_k,
            include_metadata=True,
            namespace=settings.pinecone_namespace,
        )

        candidate_matches = _extract_matches(query_result)
        candidate_count = len(candidate_matches)
        matches = _rerank_matches(request.question, candidate_matches, request.top_k)
        context, citations = _build_context_from_matches(matches)

        if not context:
            response = QueryResponse(
                answer="I could not find relevant context in the current knowledge base.",
                citations=[],
            )
            if request.use_cache:
                _write_query_cache(redis_client, key, response)
            response_obj = response
            _log_query_to_mlflow(
                request,
                response,
                latency_ms=(time.perf_counter() - started_at) * 1000,
                candidate_count=candidate_count,
                status="ok",
            )
            return response

        if request.generate_answer:
            answer = _generate_answer(openai_client, request.question, context)
        else:
            answer = "Generation skipped for retrieval-focused evaluation."
        response = QueryResponse(answer=answer, citations=citations)
        if request.use_cache:
            _write_query_cache(redis_client, key, response)
        response_obj = response
        _log_query_to_mlflow(
            request,
            response,
            latency_ms=(time.perf_counter() - started_at) * 1000,
            candidate_count=candidate_count,
            status="ok",
        )
        return response
    except HTTPException:
        status = "error"
        error_type = "http_exception"
        raise
    except Exception as error:
        status = "error"
        error_type = type(error).__name__
        _log_query_to_mlflow(
            request,
            None,
            latency_ms=(time.perf_counter() - started_at) * 1000,
            candidate_count=candidate_count,
            status="error",
            error_detail=str(error),
        )
        raise HTTPException(status_code=500, detail=f"Query failed: {error}") from error
    finally:
        elapsed_seconds = max(0.0, time.perf_counter() - started_at)
        QUERY_IN_FLIGHT.dec()
        QUERY_DURATION_SECONDS.observe(elapsed_seconds)
        QUERY_TOP_K.observe(float(request.top_k))
        QUERY_RERANK_MULTIPLIER.observe(float(effective_multiplier))
        QUERY_CACHE_CHECKS_TOTAL.labels(result=cache_result).inc()

        if candidate_count >= 0:
            QUERY_CANDIDATE_MATCHES.observe(float(candidate_count))

        cache_hit_label = "false"
        if response_obj is not None:
            citation_count = float(len(response_obj.citations))
            QUERY_CITATIONS_RETURNED.observe(citation_count)

            if response_obj.citations:
                mean_score = sum(c.score for c in response_obj.citations) / max(
                    1, len(response_obj.citations)
                )
                QUERY_MEAN_CITATION_SCORE.observe(mean_score)

            cache_hit_label = "true" if response_obj.cache_hit else "false"

        QUERY_REQUESTS_TOTAL.labels(
            status=status,
            cache_hit=cache_hit_label,
            generate_answer="true" if request.generate_answer else "false",
        ).inc()

        if status == "error":
            QUERY_ERRORS_TOTAL.labels(error_type=error_type or "unknown").inc()
