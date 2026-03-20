PYTHON ?= python
PIP ?= pip

.PHONY: install-dev lint test run-api run-ui run-etl-extract run-etl-transform run-etl-chunk run-etl-load up down

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

up:
	docker compose up --build -d

down:
	docker compose down
