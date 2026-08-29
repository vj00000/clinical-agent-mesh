"""The prior-auth specialist subgraph.

Demo depth. The model reads the retrieved coverage policy and marks each criterion
against the request; `decide_coverage` computes the verdict and `render_decision`
writes it. The model never states the outcome, so a persuasive paragraph cannot
approve something the criteria deny.
"""

from collections.abc import Callable
from typing import Any, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from mesh.agents.prior_auth_rules import (
    Criterion,
    Decision,
    decide_coverage,
    render_decision,
)
from mesh.retrieval.chunking import Chunk
from mesh.state import Citation

TOP_N_CHUNKS = 5

# Provenance for a criterion citing a chunk that was never retrieved. Such a
# citation has to survive so guard_out can reject the answer, and `source` is a
# required field, so it needs a non-empty placeholder.
UNKNOWN_SOURCE = "unretrieved"

CriteriaReader = Callable[[str, list[Chunk]], list[Criterion]]


class Retriever(Protocol):
    def search(self, query: str, *, top_k: int = 20) -> list[str]: ...


class ChunkStore(Protocol):
    def fetch(self, chunk_ids: list[str]) -> list[Chunk]: ...


class Reranker(Protocol):
    def rerank(self, query: str, candidates: list[Chunk], *, top_n: int) -> list[Chunk]: ...


class PriorAuthState(TypedDict):
    query: str
    retrieved_ids: list[str]
    chunks: list[Chunk]
    criteria: list[Criterion]
    decision: Decision
    answer: str
    citations: list[Citation]
    route: str


def build_prior_auth_subgraph(
    *,
    retriever: Retriever,
    store: ChunkStore,
    reranker: Reranker,
    read_criteria: CriteriaReader,
) -> Any:
    """Compile the prior-auth subgraph around its injected dependencies."""

    def retrieve(state: PriorAuthState) -> dict[str, Any]:
        # No try/except: RetrievalUnavailable propagates. Deciding coverage
        # against a corpus that failed to load is worse than not deciding.
        return {"retrieved_ids": retriever.search(state["query"])}

    def rerank(state: PriorAuthState) -> dict[str, Any]:
        candidates = store.fetch(state["retrieved_ids"])
        return {"chunks": reranker.rerank(state["query"], candidates, top_n=TOP_N_CHUNKS)}

    def check_criteria(state: PriorAuthState) -> dict[str, Any]:
        # Criteria citing an unretrieved chunk are kept, not filtered: dropping a
        # failed criterion could flip a denial into an approval. guard_out
        # rejects the whole answer instead.
        return {"criteria": read_criteria(state["query"], state["chunks"])}

    def decide(state: PriorAuthState) -> dict[str, Any]:
        criteria = state["criteria"]
        decision = decide_coverage(criteria)
        source_by_id = {chunk.chunk_id: chunk.source for chunk in state["chunks"]}

        citations = [
            Citation(
                chunk_id=criterion.chunk_id,
                source=source_by_id.get(criterion.chunk_id, UNKNOWN_SOURCE),
                quote=criterion.text,
            )
            for criterion in criteria
        ]

        update: dict[str, Any] = {
            "decision": decision,
            "answer": render_decision(decision, criteria),
            "citations": citations,
        }

        # more_info asks rather than asserts, so it is routed as a clarifying
        # question -- the same exemption the triage follow-up uses.
        if decision is Decision.MORE_INFO:
            update["route"] = "clarify"

        return update

    builder = StateGraph(PriorAuthState)
    builder.add_node("retrieve", retrieve)
    builder.add_node("rerank", rerank)
    builder.add_node("check_criteria", check_criteria)
    builder.add_node("decide", decide)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "rerank")
    builder.add_edge("rerank", "check_criteria")
    builder.add_edge("check_criteria", "decide")
    builder.add_edge("decide", END)

    return builder.compile()
