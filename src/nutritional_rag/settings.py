from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    project_name: str = "Nutritional RAG"
    app_env: str = "development"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4"
    openai_embedding_model: str = "text-embedding-3-small"
    pinecone_api_key: str | None = None
    pinecone_index: str | None = None
    pinecone_namespace: str = "nutrition"
    redis_url: str = "redis://localhost:6379/0"
    query_cache_ttl_seconds: int = 600
    rerank_candidate_multiplier: int = 3
    mlflow_tracking_uri: str = "http://localhost:5001"
    mlflow_experiment_name: str = "nutritional-rag-query"
    mlflow_log_queries: bool = True

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
