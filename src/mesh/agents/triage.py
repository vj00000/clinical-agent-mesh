"""The triage specialist subgraph.

Demo depth, and disclosed as such: two urgency levels and a hand-written rule
table, not a validated protocol such as the Manchester Triage System.

Two things here are deliberately not the model's decision. The red-flag rules set a
floor it can raise but never lower, and the follow-up question fires on a word count
rather than on the model choosing to ask one.

The spec calls for a LangGraph `interrupt` here. A genuine interrupt needs the
thread_id sessions and Postgres checkpointer that arrive with the API, so for now
the follow-up is returned as a clarifying question and the next turn carries the
detail. Recorded in DECISIONS.md.
"""

from collections.abc import Callable
from typing import Any, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from mesh.agents.triage_rules import (
    Urgency,
    apply_urgency_floor,
    assess_urgency,
    needs_more_detail,
)
from mesh.guardrails.nodes import SAFETY_ESCALATION_FLAG
from mesh.retrieval.chunking import Chunk
from mesh.state import Citation

TOP_N_CHUNKS = 5

EMERGENCY_BANNER = (
    "Based on what you have described, this needs emergency care now. Call your "
    "local emergency number or go to an emergency department. Please read the rest "
    "of this only once you have done that."
)

FOLLOW_UP_QUESTION = (
    "I need a little more to go on. When did this start, how bad is it, and is "
    "anything else happening alongside it?"
)

# (answer, citations, the model's own urgency read)
Adviser = Callable[[str, list[Chunk]], tuple[str, list[Citation], Urgency]]


class Retriever(Protocol):
    def search(self, query: str, *, top_k: int = 20) -> list[str]: ...


class ChunkStore(Protocol):
    def fetch(self, chunk_ids: list[str]) -> list[Chunk]: ...


class Reranker(Protocol):
    def rerank(self, query: str, candidates: list[Chunk], *, top_n: int) -> list[Chunk]: ...


class TriageState(TypedDict):
    query: str
    matched: list[str]
    floor: Urgency
    level: Urgency
    retrieved_ids: list[str]
    chunks: list[Chunk]
    answer: str
    citations: list[Citation]
    route: str
    guard_flags: list[str]


def build_triage_subgraph(
    *,
    retriever: Retriever,
    store: ChunkStore,
    reranker: Reranker,
    advise: Adviser,
) -> Any:
    """Compile the triage subgraph around its injected dependencies."""

    def assess(state: TriageState) -> dict[str, Any]:
        assessment = assess_urgency(state["query"])
        return {
            "matched": assessment.matched,
            "floor": assessment.level,
            "retrieved_ids": [],
            "guard_flags": [],
        }

    def ask_followup(state: TriageState) -> dict[str, Any]:
        # Routed as a clarifying question so guard_out does not demand citations:
        # it asserts nothing clinical, it asks for more.
        return {
            "answer": FOLLOW_UP_QUESTION,
            "citations": [],
            "route": "clarify",
            "level": state["floor"],
        }

    def retrieve(state: TriageState) -> dict[str, Any]:
        return {"retrieved_ids": retriever.search(state["query"])}

    def rerank(state: TriageState) -> dict[str, Any]:
        candidates = store.fetch(state["retrieved_ids"])
        return {"chunks": reranker.rerank(state["query"], candidates, top_n=TOP_N_CHUNKS)}

    def advise_node(state: TriageState) -> dict[str, Any]:
        answer, citations, model_level = advise(state["query"], state["chunks"])
        level = apply_urgency_floor(model_level, floor=state["floor"])

        if level is not Urgency.EMERGENCY:
            return {"answer": answer, "citations": citations, "level": level}

        # The banner is a fixed string, never model output: the one sentence that
        # matters most is the one that must not vary between runs.
        return {
            "answer": f"{EMERGENCY_BANNER}\n\n{answer}",
            "citations": citations,
            "level": level,
            "guard_flags": [SAFETY_ESCALATION_FLAG],
        }

    def _after_assess(state: TriageState) -> str:
        if needs_more_detail(state["query"], matched=state["matched"]):
            return "ask_followup"
        return "retrieve"

    builder = StateGraph(TriageState)
    builder.add_node("assess", assess)
    builder.add_node("ask_followup", ask_followup)
    builder.add_node("retrieve", retrieve)
    builder.add_node("rerank", rerank)
    builder.add_node("advise", advise_node)

    builder.add_edge(START, "assess")
    builder.add_conditional_edges(
        "assess", _after_assess, {"ask_followup": "ask_followup", "retrieve": "retrieve"}
    )
    builder.add_edge("ask_followup", END)
    builder.add_edge("retrieve", "rerank")
    builder.add_edge("rerank", "advise")
    builder.add_edge("advise", END)

    return builder.compile()
