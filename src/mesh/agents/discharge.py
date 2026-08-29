"""The discharge specialist subgraph.

Demo depth. Three things deliberately run outside the model: interactions come
from a tool call rather than from recall, the reading grade is computed rather
than judged, and the interaction warnings are appended verbatim rather than
paraphrased.

A tool result is evidence too, so each interaction carries the id of the record it
came from and joins `retrieved_ids`. Without that, guard_out would see warnings no
retrieved chunk supports and refuse the whole answer.
"""

from collections.abc import Callable
from typing import Any, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from mesh.readability import flesch_kincaid_grade
from mesh.retrieval.chunking import Chunk
from mesh.state import Citation

TOP_N_CHUNKS = 5

# Patient-facing text. US discharge guidance commonly targets grade 6-8; 8 is the
# lenient end of that range, and still far below the grade 12+ a drug label reads at.
READING_GRADE_TARGET = 8.0

# One pass, then accept what we have. A second rewrite of already-simplified text
# tends to drop clinical detail rather than shorten sentences.
MAX_SIMPLIFICATIONS = 1

INTERACTION_HEADING = "Check these combinations with your pharmacist before taking them:"

SIMPLIFY_PROMPT = (
    "Rewrite the following discharge instructions for a reader at about a US grade 8 "
    "level. Keep every dose, timing, and warning exactly as it is; shorten sentences "
    "and replace clinical words with plain ones.\n\n"
)


class Medication(BaseModel):
    name: str = Field(min_length=1)
    dose: str = ""
    frequency: str = ""


class Interaction(BaseModel):
    """One interaction warning from the drug label data.

    `chunk_id` is the record it came from, so the warning is citable like any
    other piece of evidence rather than an uncited assertion.
    """

    drugs: list[str] = Field(min_length=2)
    warning: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    source: str = Field(min_length=1)


MedicationExtractor = Callable[[str], list[Medication]]
InteractionLookup = Callable[[list[str]], list[Interaction]]

# Same shape as the guideline drafter, so `build_drafter` satisfies it unchanged.
Drafter = Callable[[str, list[Chunk]], tuple[str, list[Citation]]]


class Retriever(Protocol):
    def search(self, query: str, *, top_k: int = 20) -> list[str]: ...


class ChunkStore(Protocol):
    def fetch(self, chunk_ids: list[str]) -> list[Chunk]: ...


class Reranker(Protocol):
    def rerank(self, query: str, candidates: list[Chunk], *, top_n: int) -> list[Chunk]: ...


class DischargeState(TypedDict):
    query: str
    medications: list[Medication]
    interactions: list[Interaction]
    retrieved_ids: list[str]
    chunks: list[Chunk]
    answer: str
    citations: list[Citation]
    reading_grade: float
    simplifications: int


def build_discharge_subgraph(
    *,
    extract_medications: MedicationExtractor,
    lookup_interactions: InteractionLookup,
    retriever: Retriever,
    store: ChunkStore,
    reranker: Reranker,
    draft: Drafter,
) -> Any:
    """Compile the discharge subgraph around its injected dependencies."""

    def extract(state: DischargeState) -> dict[str, Any]:
        return {
            "medications": extract_medications(state["query"]),
            "simplifications": 0,
        }

    def lookup(state: DischargeState) -> dict[str, Any]:
        names = [medication.name for medication in state["medications"]]

        # One medication cannot interact with anything, so the tool is not called.
        # A network round trip guaranteed to return nothing is still a round trip.
        return {"interactions": lookup_interactions(names) if len(names) > 1 else []}

    def retrieve(state: DischargeState) -> dict[str, Any]:
        names = [medication.name for medication in state["medications"]]

        # Search the drug names when extraction found any: the label text is
        # indexed under the drug, not under the phrasing of the question.
        return {"retrieved_ids": retriever.search(" ".join(names) if names else state["query"])}

    def rerank(state: DischargeState) -> dict[str, Any]:
        candidates = store.fetch(state["retrieved_ids"])
        return {"chunks": reranker.rerank(state["query"], candidates, top_n=TOP_N_CHUNKS)}

    def draft_instructions(state: DischargeState) -> dict[str, Any]:
        answer, citations = draft(state["query"], state["chunks"])
        return {"answer": answer, "citations": citations}

    def score(state: DischargeState) -> dict[str, Any]:
        return {"reading_grade": flesch_kincaid_grade(state["answer"])}

    def simplify(state: DischargeState) -> dict[str, Any]:
        answer, citations = draft(f"{SIMPLIFY_PROMPT}{state['answer']}", state["chunks"])
        return {
            "answer": answer,
            "citations": citations,
            "simplifications": state["simplifications"] + 1,
        }

    def merge_interactions(state: DischargeState) -> dict[str, Any]:
        """Append the tool's warnings after scoring, verbatim.

        After scoring on purpose: the warnings are label text, not ours to
        simplify, and letting them drag the grade up would trigger a rewrite of
        the instructions to compensate for prose we did not write.
        """
        interactions = state["interactions"]
        if not interactions:
            return {}

        lines = [state["answer"], "", INTERACTION_HEADING]
        lines.extend(
            f"- {' + '.join(interaction.drugs)}: {interaction.warning}"
            for interaction in interactions
        )

        return {
            "answer": "\n".join(lines),
            "citations": [
                *state["citations"],
                *(
                    Citation(
                        chunk_id=interaction.chunk_id,
                        source=interaction.source,
                        quote=interaction.warning,
                    )
                    for interaction in interactions
                ),
            ],
            "retrieved_ids": [
                *state["retrieved_ids"],
                *(interaction.chunk_id for interaction in interactions),
            ],
        }

    def _after_score(state: DischargeState) -> str:
        too_hard = state["reading_grade"] > READING_GRADE_TARGET
        may_retry = state["simplifications"] < MAX_SIMPLIFICATIONS

        return "simplify" if too_hard and may_retry else "merge_interactions"

    builder = StateGraph(DischargeState)
    builder.add_node("extract", extract)
    builder.add_node("lookup", lookup)
    builder.add_node("retrieve", retrieve)
    builder.add_node("rerank", rerank)
    builder.add_node("draft", draft_instructions)
    builder.add_node("score", score)
    builder.add_node("simplify", simplify)
    builder.add_node("merge_interactions", merge_interactions)

    builder.add_edge(START, "extract")
    builder.add_edge("extract", "lookup")
    builder.add_edge("lookup", "retrieve")
    builder.add_edge("retrieve", "rerank")
    builder.add_edge("rerank", "draft")
    builder.add_edge("draft", "score")
    builder.add_conditional_edges(
        "score",
        _after_score,
        {"simplify": "simplify", "merge_interactions": "merge_interactions"},
    )
    builder.add_edge("simplify", "score")
    builder.add_edge("merge_interactions", END)

    return builder.compile()
