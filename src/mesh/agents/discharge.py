"""The discharge specialist subgraph.

Demo depth. The model extracts the medication list and writes the instructions;
the interaction lookup and the readability gate are code, so neither depends on
the model remembering to do them.

extract -> retrieve -> rerank -> check interactions -> draft -> score ->
                                                                        |
                                             simplify <---------------+
                                                                        |
                                               attach_interactions <----+ -> END


Known limitation: interaction warnings come from the drug-label tool, not from 
the vector store, so they carry no retrived chunk id and guard_out does not 
verify them. The tool result has its own provenve; a production version would
make that a first-class citation source rather than a special case.
"""

from collections.abc import Callable
from typing import Any, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from mesh.agents._discharge_rules import (
    MAX_READING_GRADE,
    Interaction,
    Medication,
    flesch_kincaid_grade,
    render_interactions,
)
from mesh.retrieval.chunking import Chunk
from mesh.state import Citation

TOP_N_CHUNKS = 5

# Bounded like the guideline revise loop, but it ends differently: see
# '_after_score'.
MAX_SIMPLIFY_PASSES = 2

Extractor = Callable[[str], list[Medication]]
InteractionLookup = Callable[[list[str]], list[Interaction]]
Drafter = Callable[[str, list[Chunk]], tuple[str, list[Citation]]]
Simplifier = Callable[[str], str]

class Retriver(Protocol):
    def search(self, query: str, *, top_k: int = 20) -> list[str]: ...

class ChunkStore(Protocol):
    def fetch(self, chunk_ids: list[str]) -> list[Chunk]: ...

class Reranker(Protocol):
    def rerank(self, query: str, candidates: list[Chunk], *, top_n: int) -> list[Chunk]: ...

class DischargeState(TypedDict):
    query: str
    Medications: list[Medication]
    interactions: list[Interaction]
    retrieved_ids: list[str]
    chunks: list[Chunk]
    answer: str
    citations: list[Citation]
    grade: float
    simplify_count: int

def build_discharge_subgraph(
    *,
    retriever: Retriver,
    store: ChunkStore,
    reranker: Reranker,
    extract: Extractor,
    lookup_interactions: InteractionLookup,
    draft: Drafter,
    simplify: Simplifier,
) -> Any:
    """Compile the discharge subgraph around its injected dependencies."""

    def extract_medications(state: DischargeState) -> dict[str, Any]:
        return {"Medications": extract(state["query"]), "simplify_count": 0}

    def retrieve(state: DischargeState) -> dict[str, Any]:
        return {"retrieved_ids": retriever.search(state["query"])}

    def rerank(state: DischargeState) -> dict[str, Any]:
        candidates = store.fetch(state["retrieved_ids"])
        return {"chunks": reranker.rerank(state["query"], candidates, top_n=TOP_N_CHUNKS)}

    def check_interactions(state: DischargeState) -> dict[str, Any]:
        names = [medication.name for medication in state["Medications"]]
        if not names:
            return {"interactions": []}

        return {"interactions": lookup_interactions(names)}

    def draft_instructions(state: DischargeState) -> dict[str, Any]:
        answer, citations = draft(state["query"], state["chunks"])
        return {"answer": answer, "citations": citations}

    def score(state: DischargeState) -> dict[str, Any]:
        return {"grade": flesch_kincaid_grade(state["answer"])}

    def simplify_instructions(state: DischargeState) -> dict[str, Any]:
        return {
            "answer": simplify(state["answer"]),
            "simplify_count": state["simplify_count"] + 1,
        }

    def attach_interactions(state: DischargeState) -> dict[str, Any]:
        # Attached after the readability loop, not before: the warnings are 
        # fixed text and must not be reworded by simplify, nor drag the score
        # of the instructions the gate is actually judging.
        warnings = render_interactions(state["interactions"])
        if not warnings:
            return {}  

        return {"answer": f"{warnings}\n\n{state['answer']}"}

    def _after_score(state: DischargeState) -> str:
        if state["grade"] <= MAX_READING_GRADE:
            return "done"

        # Unlike the guideline revise loop, exhausting the budget does not
        # refuse. Prose that is a grade too dense is still usable; an 
        # ungrounded clinical claim never is.
        if state["simplify_count"] >= MAX_SIMPLIFY_PASSES:
            return "done"

        return "simplify"

    builder = StateGraph(DischargeState)
    builder.add_node("extract", extract_medications)
    builder.add_node("retrieve", retrieve)
    builder.add_node("rerank", rerank)
    builder.add_node("check_interactions", check_interactions)
    builder.add_node("draft", draft_instructions)
    builder.add_node("score", score)
    builder.add_node("simplify", simplify_instructions)
    builder.add_node("attach_interactions", attach_interactions)

    builder.add_edge(START, "extract")
    builder.add_edge("extract", "retrieve")
    builder.add_edge("retrieve", "rerank")
    builder.add_edge("rerank", "check_interactions")
    builder.add_edge("check_interactions", "draft")
    builder.add_edge("draft", "score")
    builder.add_conditional_edges(
        "score", _after_score, {"simplify": "simplify", "done": "attach_interactions"}
    )
    builder.add_edge("simplify", "score")
    builder.add_edge("attach_interactions", END)

    return builder.compile()
