PYTHON ?= python
PIP ?= pip

.PHONY: install-dev lint test run-api run-ui run-etl-extract run-etl-transform run-etl-chunk run-etl-load run-eval run-eval-ragas run-eval-sweep run-demo-traffic up down

install-dev:
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[api,ui,dev]"

lint:
	ruff check .

test:
	pytest

run-api:
	uvicorn apps.api.main:app --reload --host 0.0.0.0 --port 8000

run-ui:
	streamlit run apps/ui/Home.py

run-etl-extract:
	PYTHONPATH=src $(PYTHON) -m nutritional_rag.etl.cli --stage extract --config etl/sources.bodybuilding.example.json

run-etl-transform:
	PYTHONPATH=src $(PYTHON) -m nutritional_rag.etl.cli --stage transform --input data/raw/extracted_documents.ndjson --nutrition-only --min-nutrition-score 2

run-etl-chunk:
	PYTHONPATH=src $(PYTHON) -m nutritional_rag.etl.cli --stage chunk --input data/processed/bodybuilding_book_transformed_nutrition_only.ndjson --output data/processed/chunks.ndjson

run-etl-load:
	PYTHONPATH=src $(PYTHON) -m nutritional_rag.etl.cli --stage load --input data/processed/chunks.ndjson --batch-size 100

run-eval:
	$(PYTHON) scripts/evaluate_rag.py --api-base-url http://127.0.0.1:8001 --eval-set data/eval/nutrition_eval_set.ndjson

run-eval-ragas:
	$(PYTHON) scripts/evaluate_rag.py --api-base-url http://127.0.0.1:8001 --eval-set data/eval/nutrition_eval_set.ndjson

run-eval-sweep:
	$(PYTHON) scripts/sweep_eval.py --api-base-url http://127.0.0.1:8001 --eval-set data/eval/nutrition_eval_set.ndjson --top-k-values 3,5,8 --rerank-candidate-multipliers 1,2,3

run-demo-traffic:
	bash scripts/demo_grafana_traffic.sh

up:
	docker compose up --build -d

down:
	docker compose down
