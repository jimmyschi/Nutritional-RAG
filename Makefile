PYTHON ?= python
PIP ?= pip

.PHONY: install-dev lint test run-api run-ui up down

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

up:
	docker compose up --build -d

down:
	docker compose down
