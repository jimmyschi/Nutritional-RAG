# Nutritional RAG

Initial repository scaffold for a nutritional retrieval-augmented generation system built around this stack:

- LangChain for orchestration
- Pinecone for vector retrieval
- OpenAI embeddings with `text-embedding-3-small`
- GPT-4 for generation
- PyTorch plus sentence-transformers for reranking
- MLflow for experiment tracking and registry
- FastAPI for serving
- Redis for caching
- Prometheus and Grafana for metrics and dashboards
- Streamlit for the demo UI
- Docker Compose for local orchestration
- GitHub Actions for CI and container publishing

## Repo Layout

```text
.
├── .github/workflows/
├── apps/
│   ├── api/
│   └── ui/
├── data/
│   ├── eval/
│   ├── processed/
│   └── raw/
├── ml/
│   └── reranker/
├── notebooks/
├── src/
│   └── nutritional_rag/
└── tests/
```

## What Is Included

- Python project metadata in `pyproject.toml`
- Minimal FastAPI app with health and readiness endpoints
- Minimal Streamlit landing page for the demo UI
- Make targets for local development
- GitHub Actions CI workflow for linting and tests
- GitHub Actions publish workflow for API and UI images to GHCR

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[api,ui,dev]"
cp .env.example .env
make test
make run-api
```

In a second shell:

```bash
source .venv/bin/activate
make run-ui
```

## Local Infra with Docker Compose

Start the full local stack (API, UI, Redis, MLflow, Prometheus, Grafana):

```bash
cp .env.example .env
docker compose up --build -d
```

Useful endpoints:

- API: `http://localhost:8000`
- API metrics: `http://localhost:8000/metrics`
- UI: `http://localhost:8501`
- Redis: `localhost:6379`
- MLflow: `http://localhost:5001`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`

Stop services:

```bash
docker compose down
```

## GitHub Actions

- `ci.yml` runs Ruff and pytest on pushes and pull requests
- `docker-publish.yml` builds and publishes the API and UI images to GHCR on pushes to `main`, version tags, and manual dispatch

## Next Build Steps

1. Add ingestion and chunking pipelines for nutritional data.
2. Wire Pinecone indexing and retrieval through LangChain.
3. Add the reranker training and serving path under `ml/reranker`.
4. Wire retrieval, reranking, and answer generation into the API and UI.
