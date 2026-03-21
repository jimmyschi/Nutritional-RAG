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
- Starter ETL extract pipeline modules for CSV, JSON, HTML, and text sources
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

Run the starter extract pipeline:

```bash
source .venv/bin/activate
export PYTHONPATH=src
make run-etl-extract
```

This writes normalized raw documents to `data/raw/extracted_documents.ndjson`.

For real local sources (recommended):

```bash
cp etl/sources.bodybuilding.example.json etl/sources.bodybuilding.local.json
# edit the location path in the local file
python -m nutritional_rag.etl.cli --stage extract --config etl/sources.bodybuilding.local.json
```

Run the transform stage:

```bash
source .venv/bin/activate
export PYTHONPATH=src
make run-etl-transform
```

This writes transformed records to `data/processed/transformed_documents.ndjson`.
By default, this stage applies nutrition-only filtering with keyword/rule scoring.

Run the chunk stage:

```bash
source .venv/bin/activate
export PYTHONPATH=src
make run-etl-chunk
```

This writes chunked records to `data/processed/chunks.ndjson`.

Run the load stage (OpenAI embeddings + Pinecone upsert):

```bash
source .venv/bin/activate
export PYTHONPATH=src
# Uses OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX from .env
make run-etl-load
```

Use dry-run for local validation without external API calls:

```bash
python -m nutritional_rag.etl.cli --stage load --input data/processed/chunks.ndjson --dry-run
```

Query your loaded knowledge base:

```bash
source .venv/bin/activate
export PYTHONPATH=src
uvicorn apps.api.main:app --reload --host 0.0.0.0 --port 8000
```

Then in another shell:

```bash
curl -X POST http://localhost:8000/query \
	-H "Content-Type: application/json" \
	-d '{"question":"How are carbohydrates used in endurance exercise?","top_k":5}'
```

## MLflow Query Tracking and Evaluation

Query-time MLflow logging is supported directly in the API path (best-effort, non-blocking).

Environment variables:

- `MLFLOW_TRACKING_URI` (default: `http://localhost:5001`)
- `MLFLOW_EXPERIMENT_NAME` (default: `nutritional-rag-query`)
- `MLFLOW_LOG_QUERIES` (default: `true`)

Install ML dependencies if needed:

```bash
pip install -e ".[ml]"
```

Run a lightweight evaluation set against the live API and log aggregated metrics to MLflow:

```bash
python scripts/evaluate_rag.py \
	--api-base-url http://127.0.0.1:8001 \
	--eval-set data/eval/nutrition_eval_set.ndjson \
  --rerank-candidate-multiplier 3 \
  --skip-generation \
  --mlflow-tracking-uri http://localhost:5001 \
  --mlflow-experiment nutritional-rag-eval
```

Or use:

```bash
make run-eval
```

Run a parameter sweep and log one MLflow run per configuration:

```bash
python scripts/sweep_eval.py \
  --api-base-url http://127.0.0.1:8001 \
  --eval-set data/eval/nutrition_eval_set.ndjson \
  --top-k-values 3,5,8 \
  --rerank-candidate-multipliers 1,2,3 \
  --mlflow-tracking-uri http://localhost:5001 \
  --mlflow-experiment nutritional-rag-sweep
```

The sweep script runs in retrieval-only mode so you can compare retrieval latency and citation quality
without waiting for full answer generation on every configuration.

Or use:

```bash
make run-eval-sweep
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

Provisioned Grafana dashboard:

- Folder: `Nutritional RAG`
- Dashboard: `Nutritional RAG - Query Overview`

Grafana default login (from compose env defaults):

- Username: `admin`
- Password: `admin`

If login fails because a prior password was persisted in Docker volume state,
reset Grafana state with:

```bash
docker compose down -v
docker compose up --build -d
```

If you add or change dashboard provisioning files and do not see updates immediately:

```bash
docker compose restart grafana prometheus api
```

## Prometheus Metrics (Query Path)

The API exports these custom metrics at `/metrics`:

- `nutritional_rag_query_requests_total{status,cache_hit,generate_answer}`
- `nutritional_rag_query_errors_total{error_type}`
- `nutritional_rag_query_cache_checks_total{result}` where result is `hit`, `miss`, or `skipped`
- `nutritional_rag_query_in_flight`
- `nutritional_rag_query_duration_seconds` (histogram)
- `nutritional_rag_query_top_k` (histogram)
- `nutritional_rag_query_rerank_candidate_multiplier` (histogram)
- `nutritional_rag_query_candidate_matches` (histogram)
- `nutritional_rag_query_citations_returned` (histogram)
- `nutritional_rag_query_mean_citation_score` (histogram)

Stop services:

```bash
docker compose down
```

## Real Source Testing

You can test extraction against real local data sources, including PDF books.

Example PDF source config:

- `etl/sources.bodybuilding.example.json` (template)

Use your local-only config (ignored by git):

```bash
cp etl/sources.bodybuilding.example.json etl/sources.bodybuilding.local.json
# edit location to your absolute local PDF path

source .venv/bin/activate
pip install -e ".[dev]"
export PYTHONPATH=src
python -m nutritional_rag.etl.cli --stage extract --config etl/sources.bodybuilding.local.json
python -m nutritional_rag.etl.cli --stage transform --input data/raw/extracted_documents.ndjson
```

For PDFs, extraction currently emits one document per page with page metadata,
which is a good base for later nutrition-only filtering and chunking.

## GitHub Actions

- `ci.yml` runs Ruff and pytest on pushes and pull requests
- `docker-publish.yml` builds and publishes the API and UI images to GHCR on pushes to `main`, version tags, and manual dispatch

## Next Build Steps

1. Expand source adapters and add production-grade extraction quality checks.
2. Add transformation and chunking pipelines for nutritional records.
3. Wire Pinecone indexing and retrieval through LangChain.
4. Add the reranker training and serving path under `ml/reranker`.
5. Wire retrieval, reranking, and answer generation into the API and UI.
6. Evaluate `PydanticAI` in a later stage if we add agentic LLM workflows.
