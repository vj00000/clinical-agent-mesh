.PHONY: help install up down logs test lint types check ingest eval

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-10s %s\n", $$1, $$2}'

install:  ## Install dependencies into .venv
	uv sync --extra postgres --extra observability

up:  ## Start chroma + postgres
	@command -v docker >/dev/null 2>&1 || { echo "Docker CLI not found. Install Docker Desktop, then rerun 'make up'."; exit 1; }
	@if ! docker info >/dev/null 2>&1; then \
		if [ "$$(uname -s)" = "Darwin" ] && command -v open >/dev/null 2>&1; then \
			echo "Docker daemon is not running. Starting Docker Desktop..."; \
			open -a Docker >/dev/null 2>&1 || true; \
			for _ in $$(seq 1 30); do \
				if docker info >/dev/null 2>&1; then break; fi; \
				sleep 2; \
			done; \
		fi; \
		docker info >/dev/null 2>&1 || { echo "Docker daemon is not running. Start Docker Desktop, then rerun 'make up'."; exit 1; }; \
	fi
	docker compose up -d

down:  ## Stop services (volumes are preserved)
	docker compose down

logs:  ## Tail service logs
	docker compose logs -f

test:  ## Fast tests: no LLM calls, no network
	uv run pytest

test-network:  ## Tests that call the live public clinical APIs
	uv run pytest -m network

test-rerank:  ## Tests the real cross-encoder (needs: uv sync --extra rerank)
	uv run pytest -m rerank

lint:  ## Lint and format check
	uv run ruff check .

types:  ## Strict type check
	uv run mypy src

check: lint types test  ## Everything CI runs on a pull request

ingest:  ## Download corpora and build the indexes
	uv run python -m mesh.retrieval.ingest

eval:  ## Slow, CI-gated evaluation suite
	uv run pytest -m eval

eval-routing:  ## Score the labelled routing benchmark (one LLM call per case)
	uv run python -m mesh.evals.routing

build-guideline:  ## Run the failing guideline-subgraph tests (your build)
	uv run pytest tests/nodes/test_guideline.py -x -m todo
