from fastapi import FastAPI
from prometheus_client import make_asgi_app

from nutritional_rag.settings import settings

app = FastAPI(title=settings.project_name, version="0.1.0")
app.mount("/metrics", make_asgi_app())


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
