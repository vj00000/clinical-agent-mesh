"""HTTP surface: one JSON endpoint, one SSE endpoint, one health check.

The mesh is injected into `create_app`, so the routing, validation, and event
shaping are testable with a stub and no key. `build_app` is the production
wiring and is the only part that reads Settings.

What SSE streams here is node-level progress, not tokens. The graph runs a node
at a time, and a specialist's answer only exists once its draft node has
finished, so there is nothing to emit token by token without pushing streaming
down into every model call. Calling it token streaming would be a lie; watching
`guard_in -> supervisor -> guideline -> guard_out` arrive live is still the
thing that makes the mesh legible from outside.
"""

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from mesh.state import Citation, new_state


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    # Names the conversation for the checkpointer. Optional: without one the
    # turn is answered and not persisted.
    thread_id: str | None = None


class AskResponse(BaseModel):
    answer: str
    route: str | None
    confidence: float
    citations: list[Citation]
    guard_flags: list[str]
    trace_id: str


def _config(thread_id: str | None) -> dict[str, Any] | None:
    return {"configurable": {"thread_id": thread_id}} if thread_id else None


def _to_response(result: dict[str, Any]) -> AskResponse:
    return AskResponse(
        answer=result["answer"],
        route=result["route"],
        confidence=result["confidence"],
        citations=result["citations"],
        guard_flags=result["guard_flags"],
        trace_id=result["trace_id"],
    )


def create_app(mesh: Any) -> FastAPI:
    """Build the app around an already-constructed mesh."""
    app = FastAPI(title="Clinical Agent Mesh", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        result = mesh.invoke(new_state(request.query), _config(request.thread_id))

        return _to_response(result)

    @app.post("/ask/stream")
    async def ask_stream(request: AskRequest) -> EventSourceResponse:
        async def events() -> AsyncIterator[dict[str, str]]:
            final: dict[str, Any] = {}

            async for update in mesh.astream(new_state(request.query), _config(request.thread_id)):
                for node, changes in update.items():
                    final.update(changes)
                    yield {"event": "node", "data": json.dumps({"node": node})}

            yield {
                "event": "answer",
                "data": _to_response(final).model_dump_json(),
            }

        return EventSourceResponse(events())

    return app


def build_app() -> FastAPI:
    """Production wiring. Reads Settings, so importing this module stays cheap."""
    import httpx

    from mesh.composition import build_default_mesh
    from mesh.models.config import Settings
    from mesh.models.providers import build_chat_model, build_embeddings
    from mesh.retrieval.sources import InteractionNote, fetch_openfda_interaction_notes

    settings = Settings()
    # Module-level so it outlives the request: the interaction lookup runs
    # mid-graph, on every discharge query, not once at startup.
    client = httpx.Client(timeout=30.0)

    def fetch_notes(drug: str) -> list[InteractionNote]:
        return fetch_openfda_interaction_notes(drug, limit=1, client=client)

    mesh = build_default_mesh(
        settings=settings,
        model=build_chat_model(settings),
        embeddings=build_embeddings(settings),
        fetch_notes=fetch_notes,
    )

    return create_app(mesh)
