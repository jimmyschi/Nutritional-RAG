PYTHON ?= python
PIP ?= pip

.PHONY: install-dev lint test run-api run-ui run-etl-extract run-etl-transform up down

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
	PYTHONPATH=src $(PYTHON) -m nutritional_rag.etl.cli --stage extract --config etl/sources.example.json

run-etl-transform:
	PYTHONPATH=src $(PYTHON) -m nutritional_rag.etl.cli --stage transform --input data/raw/extracted_documents.ndjson

up:
	docker compose up --build -d

down:
	docker compose down
