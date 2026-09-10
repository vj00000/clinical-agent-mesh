# uv rather than pip: the lockfile is the source of truth for this project, and
# `uv sync --frozen` fails loudly if it and pyproject.toml have drifted, where
# pip would happily install something else.
FROM python:3.12-slim AS base

# uv is copied from its own published image rather than curl-installed, so the
# build needs no network beyond the registry and the version is pinned.
COPY --from=ghcr.io/astral-sh/uv:0.8.17 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Dependencies first, in their own layer: they change far less often than the
# source, so editing a node does not reinstall torch.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --extra postgres

COPY src ./src
RUN uv sync --frozen --extra postgres

# Non-root. The container needs no write access to anything it ships.
RUN useradd --create-home --uid 10001 mesh
USER mesh

EXPOSE 8000

# The rerank extra is deliberately absent: it pulls torch and ~1GB of image for
# a reordering step the composition root degrades gracefully without.
CMD ["uv", "run", "uvicorn", "mesh.api.app:build_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8000"]
