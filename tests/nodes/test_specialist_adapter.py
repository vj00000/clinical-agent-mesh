"""
The parent/child state contract.

The mesh hands specialists a full MeshState and expects three fields back. A
subgraph is invoked with only a query and returns its own private state. This
adapter is where those two shapes meet - deliberately, rather than by coincidence.
"""

from typing import Any

from mesh.graph import RETRIEVER_REFUSAL, as_specialist
from mesh.retrieval.hybrid import RetrievalUnavailable
from mesh.state import Citation, new_state


class stubSubgraph:
    """Records what it was invoked with; returns a full private state."""

    def __init__(self) -> None:
        self.seen: list[tuple[str, Any]] = []

    def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.seen.append(payload)
        return {
            "query": payload["query"],
            "sub_questions": ["private"],
            "revision_count": 1,
            "unsupported": [],
            "answer": "Thiazides are first-line.",
            "citations": [Citation(chunk_id="c1", source="cdc-htn", quote="...")],
            "retrieved_ids": ["c1", "c2"],
        }


class BrokenSubgraph:
    def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise RetrievalUnavailable("chroma is down")


class EscalatingSubgraph:
    def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "answer": "This needs emergency care now.",
            "citations": [],
            "retrieved_ids": [],
            "route": "clarify",
            "guard_flags": ["safety_escalation"],
        }


def test_the_subgraph_sees_only_the_query():
    """Passing the whole MeshState would let a specialist read guard flags and
    routing confidence that are none of its business.

    `as_specialist` returns a node function, not a compiled graph, so it is
    called directly rather than through `.invoke`.
    """
    subgraph = stubSubgraph()

    as_specialist(subgraph)(new_state("what is first-line therapy for hypertension?"))

    assert subgraph.seen == [{"query": "what is first-line therapy for hypertension?"}]


def test_only_public_fields_cross_back():
    """revision_count and sub_questions are private. If they leaked into MeshState
    the subgraph could not change its internals without breaking the parent."""
    update = as_specialist(stubSubgraph())(new_state("q"))

    assert set(update) == {"answer", "citations", "retrieved_ids"}


def test_unreachable_retrieval_becomes_a_refusal_not_a_stack_trace():
    """The subgraph refuses to answer from a degraded corpus by raising. The mesh
    is where that becomes something a user can read."""
    update = as_specialist(BrokenSubgraph())(new_state("q"))

    assert update["answer"] == RETRIEVER_REFUSAL
    assert update["citations"] == []
    assert update["route"] == "refuse"


def test_the_refusal_is_recorded_in_the_audit_trail():
    update = as_specialist(BrokenSubgraph())(new_state("q"))

    assert "retriever_unavailable" in update["guard_flags"]


def test_a_specialist_may_hand_back_a_route_and_guard_flags():
    """Both change what guard_out does. A specialist that escalates has to be able
    to say so, or the escalation is silently replaced by an ungrounded refusal."""
    state = new_state("q")
    state["guard_flags"] = ["phi:EMAIL"]

    update = as_specialist(EscalatingSubgraph())(state)

    assert update["route"] == "clarify"
    assert update["guard_flags"] == ["phi:EMAIL", "safety_escalation"]
