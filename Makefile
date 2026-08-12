.PHONY: help install up down logs test lint types check ingest eval

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-10s %s\n", $$1, $$2}'

install:  ## Install dependencies into .venv
	uv sync --extra postgres --extra observability

up:  ## Start chroma + postgres
	docker compose up -d

down:  ## Stop services (volumes are preserved)
	docker compose down

logs:  ## Tail service logs
	docker compose logs -f

test:  ## Fast tests: no LLM calls, no network
	uv run pytest

test-network:  ## Tests that call the live public clinical APIs
	uv run pytest -m network

lint:  ## Lint and format check
	uv run ruff check .

types:  ## Strict type check
	uv run mypy src

check: lint types test  ## Everything CI runs on a pull request

ingest:  ## Download corpora and build the indexes
	uv run python -m mesh.retrieval.ingest

eval:  ## Slow, CI-gated evaluation suite
	uv run pytest -m eval
