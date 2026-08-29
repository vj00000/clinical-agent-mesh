"""The mesh graph: guards, supervisor, and specialist subgraphs.

The supervisor and specialists are injected rather than imported, so the wiring
can be tested without an LLM and so a specialist can be swapped for a subgraph
without touching this file.
"""

from collections.abc import Mapping
from typing import Any, Protocol

from langgraph.graph import END, START, StateGraph

from mesh.guardrails.nodes import guard_in, guard_out
from mesh.retrieval.hybrid import RetrievalUnavailable
from mesh.state import MeshState


class Node(Protocol):
    """A graph node.

    Declared as a Protocol rather than a Callable alias because LangGraph's node
    type requires a parameter *named* `state`; a bare Callable's parameter is
    anonymous and fails to match.
    """

    def __call__(self, state: MeshState) -> dict[str, Any]: ...


OUT_OF_SCOPE_REFUSAL = (
    "I can only help with clinical questions grounded in the guideline, triage, "
    "coverage, and discharge sources I have access to."
)

RETRIEVER_REFUSAL = (
    "I can't reach the clinical sources right now. Rather than answer from memory, "
    "I'd rather tell you the evidence base is unavailable."
)

CLARIFY_PROMPT = (
    "I'm not confident I understood which of those you mean. Could you say a "
    "little more about what you're asking?"
)

_SPECIALIST_NAMES = ("guideline", "triage", "prior_auth", "discharge")

# The three fields every specialist returns, stated once rather than repeated in
# each one.
_SPECIALIST_FIELDS = ("answer", "citations", "retrieved_ids")


def _refuse(state: MeshState) -> dict[str, Any]:
    return {"answer": OUT_OF_SCOPE_REFUSAL, "citations": [], "route": "refuse"}


def _clarify(state: MeshState) -> dict[str, Any]:
    return {"answer": CLARIFY_PROMPT, "citations": []}


def _after_guard_in(state: MeshState) -> str:
    """Skip the supervisor entirely when guard_in has already blocked the query."""
    return "refuse" if state["route"] == "refuse" else "supervisor"


def _after_supervisor(state: MeshState) -> str:
    route = state["route"]
    return route if route in {*_SPECIALIST_NAMES, "clarify"} else "refuse"


def as_specialist(subgraph: Any) -> Node:
    """Adapt a specialist subgraph to the mesh's state contract.

    The subgraph receives only the query and returns only its public fields, so its
    private working state can change without touching the parent graph.
    """

    def specialist(state: MeshState) -> dict[str, Any]:
        try:
            result = subgraph.invoke({"query": state["query"]})
        except RetrievalUnavailable:
            # Routed to refuse so guard_out skips the citation check: a refusal
            # asserts nothing clinical, so demanding citations from it would
            # replace a usable message with a second refusal.
            return {
                "answer": RETRIEVER_REFUSAL,
                "citations": [],
                "route": "refuse",
                "guard_flags": [*state["guard_flags"], "retriever_unavailable"],
            }

        update: dict[str, Any] = {field: result[field] for field in _SPECIALIST_FIELDS}

        # A specialist may also hand back a route (a triage follow-up is a
        # clarifying question) or guard flags (a red-flag escalation). Both change
        # what guard_out does, so they have to cross the boundary. Nothing else does.
        if result.get("route"):
            update["route"] = result["route"]
        if result.get("guard_flags"):
            update["guard_flags"] = [*state["guard_flags"], *result["guard_flags"]]

        return update

    return specialist


def build_mesh(
    *,
    supervisor: Node,
    specialists: Mapping[str, Node],
) -> Any:
    """Compile the mesh. `specialists` maps each route name to its node or subgraph."""
    missing = set(_SPECIALIST_NAMES) - set(specialists)
    if missing:
        raise ValueError(f"missing specialists: {sorted(missing)}")

    builder = StateGraph(MeshState)

    builder.add_node("guard_in", guard_in)
    builder.add_node("supervisor", supervisor)
    builder.add_node("refuse", _refuse)
    builder.add_node("clarify", _clarify)
    builder.add_node("guard_out", guard_out)
    for name in _SPECIALIST_NAMES:
        builder.add_node(name, specialists[name])

    builder.add_edge(START, "guard_in")
    builder.add_conditional_edges(
        "guard_in", _after_guard_in, {"refuse": "refuse", "supervisor": "supervisor"}
    )
    builder.add_conditional_edges(
        "supervisor",
        _after_supervisor,
        {name: name for name in (*_SPECIALIST_NAMES, "clarify", "refuse")},
    )

    for name in (*_SPECIALIST_NAMES, "clarify", "refuse"):
        builder.add_edge(name, "guard_out")
    builder.add_edge("guard_out", END)

    return builder.compile()
