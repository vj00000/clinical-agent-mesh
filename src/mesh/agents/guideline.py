"""The guideline copilot subgraph.

Every dependency is injected - retriever, chunk store, reranker,
and two model calls - so the whole subgraph runs in tests without an API key,
and so a specialist can be swapped without touching 'build_mesh'.

State is private. Only 'answer', 'citations', and retrieved_ids' cross back
to these parent mesh; 'sub_questions' and 'revision_count' are this agent's business.

"""

from collections.abc import Callable
from typing import Any, Protocol, TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from mesh.guardrails.citations import verify_citations
from mesh.guardrails.nodes import UNGROUNDED_REFUSAL
from mesh.retrieval.chunking import Chunk
from mesh.state import Citation

# Handing the model 20 mixed chunks instead of 5 strong ones costs money and
# dilutes the answer.
TOP_N_CHUNKS = 5
# Bounded on purpose: an unbounded revise loop is how a single quesry cost $40.
MAX_REVISIONS = 2

Planner = Callable[[str], list[str]]
Drafter = Callable[[str, list[Chunk]], tuple[str, list[Citation]]]
ContradictionDetector = Callable[[str, list[Chunk]], str | None]


class Retriever(Protocol):
    def search(self, query: str, *, tok_k: int = 200) -> list[str]: ...


class ChunkStore(Protocol):
    def fetch(self, chunk_ids: list[str]) -> list[Chunk]: ...


class Reranker(Protocol):
    def rerank(self, query: str, candidates: list[Chunk], *, top_n: int) -> list[Chunk]: ...


class GuidelineState(TypedDict):
    query: str
    sub_questions: list[str]
    retrieved_ids: list[str]
    chunks: list[Chunk]
    answer: str
    citations: list[Citation]
    revision_count: int
    unsupported: list[str]
    grounded: bool


def build_guideline_subgraph(
    *,
    plan: Planner,
    retriever: Retriever,
    store: ChunkStore,
    reranker: Reranker,
    draft: Drafter,
    detect_contradictions: ContradictionDetector,
) -> Any:
    """Compile the guideline subgraph around its injected dependencies;"""

    def plan_query(state: GuidelineState) -> dict[str, Any]:
        # revision_count is seeded here rather than by the caller: the parent mesh
        # invokes with only 'query', and every later node this counter.
        sub_questions = plan(state["query"])
        return {"sub_questions": sub_questions, "revision_count": 0}

    def retrieve(state: GuidelineState) -> dict[str, Any]:
        ids: list[str] = []
        seen: set[str] = set()
        for sub_question in state["sub_questions"]:
            for chunk_id in retriever.search(sub_question):
                if chunk_id in seen:
                    continue
                seen.add(chunk_id)
                ids.append(chunk_id)
        return {"retrieved_ids": ids}

    def rerank(state: GuidelineState) -> dict[str, Any]:
        candidates = store.fetch(state["retrieved_ids"])
        return {"chunks": reranker.rerank(state["query"], candidates, top_n=TOP_N_CHUNKS)}

    def draft_answer(state: GuidelineState) -> dict[str, Any]:
        answer, citations = draft(state["query"], state["chunks"])
        return {"answer": answer, "citations": citations}

    def verify(state: GuidelineState) -> dict[str, Any]:
        """Reuse the guard's verifier. A second implementation here would drift."""
        verdict = verify_citations(state["citations"], retrieved=set(state["retrieved_ids"]))
        return {"grounded": verdict.grounded, "unsupported": verdict.unsupported}

    def revise(state: GuidelineState) -> dict[str, Any]:
        """Ask again, naming the chunk ids it invented, over the same evidence."""
        prompt = (
            f"{state['query']}\n\n"
            "Your previous answer cited chunks ids that were never retrieved. "
            f"{', '.join(state['unsupported'])}.\n"
            "Cite only chunk ids from the passages you were given."
        )
        answer, citations = draft(prompt, state["chunks"])
        return {
            "answer": answer,
            "citations": citations,
            "revision_count": state["revision_count"] + 1,
        }

    def refuse(state: GuidelineState) -> dict[str, Any]:
        """Every attempt cited something never retrieved, So there is no grounded
        answer to fall back to.
        Returning the best attempt so far would be
        returning an ungrounded clinical claim.
        """
        return {"answer": UNGROUNDED_REFUSAL, "citations": []}

    def contradiction_check(state: GuidelineState) -> dict[str, Any]:
        """Surface disagreements between sources instead of silently picking one.
        Runs only on the ground path: a refusal asserts nothing clinical, so there is nothing
        for sources to disagree about.
        """
        note = detect_contradictions(state["answer"], state["chunks"])
        if note is None:
            return {}
        return {"answer": f"{state['answer']}\n\nNote: {note}"}

    def _after_verify(state: GuidelineState) -> str:
        if state["grounded"]:
            return "done"
        return "refuse" if state["revision_count"] >= MAX_REVISIONS else "revise"

    builder = StateGraph(GuidelineState)
    builder.add_node("plan_query", plan_query)
    builder.add_node("retrieve", retrieve)
    builder.add_node("rerank", rerank)
    builder.add_node("draft", draft_answer)
    builder.add_node("verify", verify)
    builder.add_node("revise", revise)
    builder.add_node("refuse", refuse)
    builder.add_node("contradiction_check", contradiction_check)

    builder.add_edge(START, "plan_query")
    builder.add_edge("plan_query", "retrieve")
    builder.add_edge("retrieve", "rerank")
    builder.add_edge("rerank", "draft")
    builder.add_edge("draft", "verify")
    builder.add_conditional_edges(
        "verify",
        _after_verify,
        {"done": "contradiction_check", "revise": "revise", "refuse": "refuse"},
    )
    builder.add_edge("revise", "verify")
    builder.add_edge("refuse", END)
    builder.add_edge("contradiction_check", END)

    return builder.compile()


PLANNER_PROMPT = """You split clinical questions into sub-questions to search seperately.

A single-part question is one sub-question. - do not invent complexity. Split only when the 
question genuinely asks two things, such as a therapy choice plus dosing in renal implairment.
Never more than three."""


class QueryPlan(BaseModel):
    sub_questions: list[str] = Field(min_length=1, max_length=3)


def build_planner(model: BaseChatModel) -> Planner:
    structured = model.with_structured_output(QueryPlan)

    def plan(query: str) -> list[str]:
        result = structured.invoke(
            [("system", PLANNER_PROMPT), ("human", query)],
        )
        return QueryPlan.model_validate(result).sub_questions

    return plan


DRAFTER_PROMPT = """Answer the clinical question using only the passages provided.

Each passage is a prefixed with its chunk id. Every claim must cite the chunk id of the 
passage supporting it, and quote the sentence you relied on. If the passages do not answer
the question, say so rather than filling the gap from memory."""


class DraftCitation(BaseModel):
    """What the model returns: an id and the sentence it relied on.

    Deliberately not 'Citation' - the model supplies the id and quote, and the source is
    looked up from the chunk it names, so it cannot invent provenance it was never given.
    """

    chunk_id: str = Field(min_length=1)
    quote: str


class DraftAnswer(BaseModel):
    answer: str
    citations: list[DraftCitation]


# A citation naming a chunk that was never retrieved still has to survive as a Citation
# object so 'verify' can reject it. Dropping it here would hide the exact failure the
# revise loop exists to catch.

UNKNOWN_SOURCE = "unretrieved"


def _format_chunks(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[{chunk.chunk_id}] {chunk.text}" for chunk in chunks)


def build_drafter(model: BaseChatModel) -> Drafter:
    structured = model.with_structured_output(DraftAnswer)

    def draft(prompt: str, chunks: list[Chunk]) -> tuple[str, list[Citation]]:
        result = structured.invoke(
            [
                ("system", DRAFTER_PROMPT),
                ("human", f"Question: {prompt}\n\nPassages:\n{_format_chunks(chunks)}"),
            ],
        )
        drafted = DraftAnswer.model_validate(result)
        source_by_id = {chunk.chunk_id: chunk.source for chunk in chunks}
        citations = [
            Citation(
                chunk_id=cited.chunk_id,
                source=source_by_id.get(cited.chunk_id, UNKNOWN_SOURCE),
                quote=cited.quote,
            )
            for cited in drafted.citations
        ]
        return drafted.answer, citations

    return draft


CONTRADICTION_PROMPT = """Decide whether the passages disagree with each other on anything
the answer asserts.

Disagreement means the sources state different things - a different first-line drug, a
different thresold, a different duration. A passage being silent on a point is not a 
disagreement. If they agree, say so; do not manufacture nuance.
"""


class ContradictionCheck(BaseModel):
    sources_disagree: bool
    note: str = ""


def build_contradiction_detector(model: BaseChatModel) -> ContradictionDetector:
    structured = model.with_structured_output(ContradictionCheck)

    def detect(answer: str, chunks: list[Chunk]) -> str | None:
        result = structured.invoke(
            [
                ("system", CONTRADICTION_PROMPT),
                ("human", f"Answer: {answer}\n\nPassages:\n{_format_chunks(chunks)}"),
            ],
        )
        check = ContradictionCheck.model_validate(result)
        return check.note if check.sources_disagree and check.note else None

    return detect
