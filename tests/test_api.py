from fastapi.testclient import TestClient

import nutritional_rag.api as api_module
from nutritional_rag.api import app

client = TestClient(app)


def test_healthcheck() -> None:
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readiness_shape() -> None:
    response = client.get("/readyz")

    assert response.status_code == 200
    body = response.json()

    assert body["status"] == "ready"
    assert set(body) == {
        "status",
        "openai_configured",
        "pinecone_configured",
        "redis_configured",
        "mlflow_configured",
    }


def test_query_returns_answer_and_citations(monkeypatch) -> None:
    monkeypatch.setattr(api_module.settings, "openai_api_key", "test-openai")
    monkeypatch.setattr(api_module.settings, "pinecone_api_key", "test-pinecone")
    monkeypatch.setattr(api_module.settings, "pinecone_index", "test-index")
    monkeypatch.setattr(api_module.settings, "rerank_candidate_multiplier", 1)

    class _EmbeddingData:
        embedding = [0.1, 0.2, 0.3]

    class _EmbeddingsResponse:
        data = [_EmbeddingData()]

    class _Message:
        content = "Protein intake supports muscle repair based on the retrieved sources."

    class _Choice:
        message = _Message()

    class _ChatResponse:
        choices = [_Choice()]

    class _EmbeddingsAPI:
        def create(self, model: str, input: list[str]):
            assert model
            assert input
            return _EmbeddingsResponse()

    class _CompletionsAPI:
        def create(self, **kwargs):
            assert kwargs["model"]
            assert kwargs["messages"]
            return _ChatResponse()

    class _ChatAPI:
        completions = _CompletionsAPI()

    class _OpenAIClient:
        embeddings = _EmbeddingsAPI()
        chat = _ChatAPI()

    class _PineconeIndex:
        def query(self, **kwargs):
            assert kwargs["top_k"] == 3
            return {
                "matches": [
                    {
                        "id": "vec-1",
                        "score": 0.9,
                        "metadata": {
                            "text": "Higher protein intake can aid recovery.",
                            "source_id": "exercise-physiology-book-pdf",
                            "document_id": "doc-1",
                            "title": "Exercise Physiology",
                            "page_number": 42,
                            "chunk_index": 0,
                        },
                    }
                ]
            }

    monkeypatch.setattr(api_module, "_get_openai_client", lambda: _OpenAIClient())
    monkeypatch.setattr(api_module, "_get_pinecone_index", lambda: _PineconeIndex())

    response = client.post(
        "/query",
        json={"question": "How much protein helps recovery?", "top_k": 3},
    )
    assert response.status_code == 200

    body = response.json()
    assert "protein" in body["answer"].lower()
    assert body["cache_hit"] is False
    assert len(body["citations"]) == 1
    assert body["citations"][0]["source_id"] == "exercise-physiology-book-pdf"


def test_query_includes_pubmed_url_when_uid_present(monkeypatch) -> None:
    monkeypatch.setattr(api_module.settings, "openai_api_key", "test-openai")
    monkeypatch.setattr(api_module.settings, "pinecone_api_key", "test-pinecone")
    monkeypatch.setattr(api_module.settings, "pinecone_index", "test-index")
    monkeypatch.setattr(api_module.settings, "rerank_candidate_multiplier", 1)

    class _EmbeddingData:
        embedding = [0.1, 0.2, 0.3]

    class _EmbeddingsResponse:
        data = [_EmbeddingData()]

    class _Message:
        content = "Creatine evidence summary."

    class _Choice:
        message = _Message()

    class _ChatResponse:
        choices = [_Choice()]

    class _EmbeddingsAPI:
        def create(self, model: str, input: list[str]):
            return _EmbeddingsResponse()

    class _CompletionsAPI:
        def create(self, **kwargs):
            return _ChatResponse()

    class _ChatAPI:
        completions = _CompletionsAPI()

    class _OpenAIClient:
        embeddings = _EmbeddingsAPI()
        chat = _ChatAPI()

    class _PineconeIndex:
        def query(self, **kwargs):
            return {
                "matches": [
                    {
                        "id": "vec-pubmed",
                        "score": 0.8,
                        "metadata": {
                            "text": "Creatine supplementation may aid repeated sprint performance.",
                            "source_id": "pubmed-sports-nutrition",
                            "document_id": "doc-pubmed-1",
                            "title": "Creatine and repeated sprint performance",
                            "uid": "12345678",
                            "chunk_index": 0,
                        },
                    }
                ]
            }

    monkeypatch.setattr(api_module, "_get_openai_client", lambda: _OpenAIClient())
    monkeypatch.setattr(api_module, "_get_pinecone_index", lambda: _PineconeIndex())

    response = client.post(
        "/query",
        json={"question": "Does creatine improve sprint performance?", "top_k": 3},
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["citations"]) == 1
    assert body["citations"][0]["pubmed_url"] == "https://pubmed.ncbi.nlm.nih.gov/12345678/"


def test_query_uses_cache_when_available(monkeypatch) -> None:
    monkeypatch.setattr(api_module.settings, "openai_api_key", "test-openai")
    monkeypatch.setattr(api_module.settings, "pinecone_api_key", "test-pinecone")
    monkeypatch.setattr(api_module.settings, "pinecone_index", "test-index")

    cached = api_module.QueryResponse(
        answer="cached answer",
        citations=[api_module.Citation(vector_id="vec-1", score=0.99)],
        cache_hit=True,
    )

    monkeypatch.setattr(api_module, "_get_redis_client", lambda: object())
    monkeypatch.setattr(api_module, "_cache_key", lambda request: "cache:key")
    monkeypatch.setattr(api_module, "_read_query_cache", lambda *args: cached)

    def _fail_openai():
        raise AssertionError("OpenAI should not be called when cache hits")

    monkeypatch.setattr(api_module, "_get_openai_client", _fail_openai)

    response = client.post("/query", json={"question": "cached question", "top_k": 5})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "cached answer"
    assert body["cache_hit"] is True


def test_query_skips_cache_and_uses_request_rerank_override(monkeypatch) -> None:
    monkeypatch.setattr(api_module.settings, "openai_api_key", "test-openai")
    monkeypatch.setattr(api_module.settings, "pinecone_api_key", "test-pinecone")
    monkeypatch.setattr(api_module.settings, "pinecone_index", "test-index")
    monkeypatch.setattr(api_module.settings, "rerank_candidate_multiplier", 9)

    class _EmbeddingData:
        embedding = [0.1, 0.2, 0.3]

    class _EmbeddingsResponse:
        data = [_EmbeddingData()]

    class _Message:
        content = "override works"

    class _Choice:
        message = _Message()

    class _ChatResponse:
        choices = [_Choice()]

    class _EmbeddingsAPI:
        def create(self, model: str, input: list[str]):
            return _EmbeddingsResponse()

    class _CompletionsAPI:
        def create(self, **kwargs):
            return _ChatResponse()

    class _ChatAPI:
        completions = _CompletionsAPI()

    class _OpenAIClient:
        embeddings = _EmbeddingsAPI()
        chat = _ChatAPI()

    class _PineconeIndex:
        def query(self, **kwargs):
            assert kwargs["top_k"] == 6
            return {
                "matches": [
                    {
                        "id": "vec-1",
                        "score": 0.9,
                        "metadata": {
                            "text": "Higher protein intake can aid recovery.",
                            "source_id": "exercise-physiology-book-pdf",
                            "document_id": "doc-1",
                            "page_number": 42,
                            "chunk_index": 0,
                        },
                    }
                ]
            }

    monkeypatch.setattr(api_module, "_get_openai_client", lambda: _OpenAIClient())
    monkeypatch.setattr(api_module, "_get_pinecone_index", lambda: _PineconeIndex())

    def _fail_read_cache(*args, **kwargs):
        raise AssertionError("Cache should not be read when use_cache is false")

    def _fail_write_cache(*args, **kwargs):
        raise AssertionError("Cache should not be written when use_cache is false")

    monkeypatch.setattr(api_module, "_read_query_cache", _fail_read_cache)
    monkeypatch.setattr(api_module, "_write_query_cache", _fail_write_cache)

    response = client.post(
        "/query",
        json={
            "question": "How much protein helps recovery?",
            "top_k": 3,
            "rerank_candidate_multiplier": 2,
            "use_cache": False,
        },
    )
    assert response.status_code == 200
    assert response.json()["cache_hit"] is False


def test_rerank_matches_preserves_multiple_sources_when_available() -> None:
    matches = [
        {
            "id": "book-1",
            "score": 0.95,
            "metadata": {
                "text": "Creatine performance recovery athletes training.",
                "source_id": "exercise-physiology-book-pdf",
            },
        },
        {
            "id": "book-2",
            "score": 0.93,
            "metadata": {
                "text": "Creatine performance recovery athletes training.",
                "source_id": "exercise-physiology-book-pdf",
            },
        },
        {
            "id": "pubmed-1",
            "score": 0.70,
            "metadata": {
                "text": "Creatine supplementation performance athletes randomized trial.",
                "source_id": "pubmed-sports-nutrition",
            },
        },
    ]

    reranked = api_module._rerank_matches(
        "Does creatine supplementation help athlete performance?",
        matches,
        top_k=2,
    )

    source_ids = {match["metadata"]["source_id"] for match in reranked}
    assert source_ids == {"exercise-physiology-book-pdf", "pubmed-sports-nutrition"}


def test_query_returns_503_when_rag_settings_missing(monkeypatch) -> None:
    monkeypatch.setattr(api_module.settings, "openai_api_key", None)
    monkeypatch.setattr(api_module.settings, "pinecone_api_key", None)
    monkeypatch.setattr(api_module.settings, "pinecone_index", None)

    response = client.post("/query", json={"question": "test query"})
    assert response.status_code == 503
    assert "OPENAI_API_KEY" in response.json()["detail"]
