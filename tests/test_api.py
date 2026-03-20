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
    assert len(body["citations"]) == 1
    assert body["citations"][0]["source_id"] == "exercise-physiology-book-pdf"


def test_query_returns_503_when_rag_settings_missing(monkeypatch) -> None:
    monkeypatch.setattr(api_module.settings, "openai_api_key", None)
    monkeypatch.setattr(api_module.settings, "pinecone_api_key", None)
    monkeypatch.setattr(api_module.settings, "pinecone_index", None)

    response = client.post("/query", json={"question": "test query"})
    assert response.status_code == 503
    assert "OPENAI_API_KEY" in response.json()["detail"]
