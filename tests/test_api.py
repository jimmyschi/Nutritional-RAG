from fastapi.testclient import TestClient

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
