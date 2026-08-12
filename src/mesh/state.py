"""The state contract between the parent mesh graph and specialist subgraphs.

Specialists keep their own private working state. Only what appears in
`MeshState` crosses the boundary, so a specialist's internals can change
without touching the parent graph.
"""

import uuid
from typing import Literal, TypedDict

from pydantic import BaseModel, Field

Route = Literal["guideline", "triage", "prior_auth", "discharge", "refuse", "clarify"]


class Citation(BaseModel):
    """A claim's link back to the retrieved chunk that supports it."""

    chunk_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    quote: str


class MeshState(TypedDict):
    query: str
    route: Route | None
    confidence: float
    answer: str
    citations: list[Citation]
    trace_id: str

    # Chunk ids the retriever returned this turn. guard_out checks citations
    # against these, so a specialist cannot cite a chunk it never saw.
    retrieved_ids: list[str]

    # Audit trail of what the guards did: which PHI categories were stripped,
    # which injection patterns fired, which citations failed. Never the PHI itself.
    guard_flags: list[str]


def new_state(query: str) -> MeshState:
    """Build the initial state for one turn through the mesh."""
    return MeshState(
        query=query,
        route=None,
        confidence=0.0,
        answer="",
        citations=[],
        trace_id=str(uuid.uuid4()),
        retrieved_ids=[],
        guard_flags=[],
    )
