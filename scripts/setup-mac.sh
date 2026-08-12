#!/usr/bin/env bash
#
# Set up clinical-agent-mesh on macOS (Apple Silicon).
#
# Safe to re-run: every step checks before it acts.
#
#   chmod +x scripts/setup-mac.sh && ./scripts/setup-mac.sh
#
set -euo pipefail

BOLD=$'\033[1m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RED=$'\033[31m'; OFF=$'\033[0m'
step()  { echo; echo "${BOLD}==> $*${OFF}"; }
ok()    { echo "${GREEN}  ok${OFF} $*"; }
warn()  { echo "${YELLOW}  !${OFF} $*"; }
fail()  { echo "${RED}  x${OFF} $*"; exit 1; }

cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

step "Checking machine"
[[ "$(uname -s)" == "Darwin" ]] || fail "This script is for macOS. On Linux just run: make install"
echo "  macOS $(sw_vers -productVersion) on $(uname -m)"
[[ "$(uname -m)" == "arm64" ]] && ok "Apple Silicon" || warn "Intel Mac: everything works, but torch will be slower"

# ---------------------------------------------------------------------------
# uv — installs and manages Python itself, so no Homebrew Python needed.
# ---------------------------------------------------------------------------
step "Installing uv"
if command -v uv >/dev/null 2>&1; then
  ok "already installed ($(uv --version))"
else
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null 2>&1 || fail "uv installed but not on PATH; open a new terminal and re-run"
  ok "installed $(uv --version)"
fi

# The installer appends to the shell profile, but this shell has not re-read it.
export PATH="$HOME/.local/bin:$PATH"

step "Installing Python 3.12"
uv python install 3.12
ok "python 3.12 available to uv"

# ---------------------------------------------------------------------------
# Docker — needed for Chroma (vector store) and Postgres (checkpointer).
# ---------------------------------------------------------------------------
step "Checking Docker"
if ! command -v docker >/dev/null 2>&1; then
  warn "Docker not found."
  echo "     Install Docker Desktop for Apple Silicon:"
  echo "       https://desktop.docker.com/mac/main/arm64/Docker.dmg"
  echo "     or:  brew install --cask docker"
  echo "     Then start Docker Desktop and re-run this script."
  DOCKER_OK=false
elif ! docker info >/dev/null 2>&1; then
  warn "Docker is installed but the daemon is not running — start Docker Desktop, then re-run."
  DOCKER_OK=false
else
  ok "docker $(docker version --format '{{.Server.Version}}') running"
  DOCKER_OK=true
fi

# ---------------------------------------------------------------------------
# Dependencies.
# ---------------------------------------------------------------------------
step "Installing project dependencies"
echo "  (torch resolves to the standard macOS wheel — the CPU-index pin in"
echo "   pyproject.toml is Linux-only, since that index has no arm64 builds)"
uv sync --extra postgres --extra observability
ok "core dependencies installed"

step "Installing the rerank extra (torch, ~1GB — optional)"
if [[ "${SKIP_RERANK:-}" == "1" ]]; then
  warn "skipped (SKIP_RERANK=1). Install later with: uv sync --extra rerank"
else
  echo "  Set SKIP_RERANK=1 to skip this. Downloading may take several minutes."
  uv sync --extra postgres --extra observability --extra rerank
  ok "rerank extra installed"
fi

# ---------------------------------------------------------------------------
# Configuration.
# ---------------------------------------------------------------------------
step "Setting up .env"
if [[ -f .env ]]; then
  ok ".env already exists (leaving it alone)"
else
  cp .env.example .env
  ok "created .env from .env.example"
  warn "EDIT .env AND ADD YOUR OPENAI_API_KEY — ingestion and every LLM node need it"
fi

# ---------------------------------------------------------------------------
# Services.
# ---------------------------------------------------------------------------
if [[ "$DOCKER_OK" == "true" ]]; then
  step "Starting Chroma and Postgres"
  docker compose up -d
  echo -n "  waiting for Chroma"
  for _ in $(seq 1 40); do
    if curl -sf http://localhost:8001/api/v2/heartbeat >/dev/null 2>&1; then
      echo; ok "chroma healthy on :8001"; break
    fi
    echo -n "."; sleep 2
  done
  curl -sf http://localhost:8001/api/v2/heartbeat >/dev/null 2>&1 \
    || warn "Chroma did not come up. Check: docker compose logs chroma"
else
  step "Skipping services (Docker unavailable)"
  warn "Integration tests will skip themselves until Chroma is running."
fi

# ---------------------------------------------------------------------------
# Verify.
# ---------------------------------------------------------------------------
step "Running the test suite"
uv run pytest

step "Lint and types"
uv run ruff check .
uv run mypy src

# ---------------------------------------------------------------------------
step "Done"
cat <<EOF

  Project:  $PROJECT_ROOT

  Next steps:
    1. Add your key:      \$EDITOR .env        # OPENAI_API_KEY=sk-...
    2. Build the corpus:  make ingest
    3. Score routing:     make eval-routing

  Useful:
    make check          lint + types + fast tests
    make test-network   tests against live PubMed / MedlinePlus
    make test-rerank    tests the real cross-encoder (needs the rerank extra)
    make up / make down start / stop Chroma and Postgres

  If 'uv' is not found in a new terminal, add this to ~/.zshrc:
    export PATH="\$HOME/.local/bin:\$PATH"

EOF
