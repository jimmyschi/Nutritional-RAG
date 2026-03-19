PYTHON ?= python
PIP ?= pip

.PHONY: install-dev lint test run-api run-ui run-etl-extract up down

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
	PYTHONPATH=src $(PYTHON) -m nutritional_rag.etl.cli --config etl/sources.example.json

up:
	docker compose up --build -d

down:
	docker compose down
