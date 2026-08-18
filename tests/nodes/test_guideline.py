"""The guideline copilot subgraph.

YOUR BUILD. These tests are written and failing; make them pass one at a time,
top to bottom. Read docs/GUIDELINE-SUBGRAPH-BRIEF.md first.

    uv run pytest tests/nodes/test_guideline.py -x

Every dependency is injected, so none of this needs an API key. The stubs below are
small real objects rather than mocks — assertions are about what the subgraph did,
never about how a mock was called.
"""

from typing import Any

import pytest

from mesh.retrieval.chunking import Chunk
from mesh.retrieval.hybrid import RetrievalUnavailable
from mesh.state import Citation

# Excluded from the default run so `make check` stays green while this is
# unimplemented. Run them with `make build-guideline`.
# pytestmark = pytest.mark.todo

# uncomment above line to remove from default test suite

# --- stubs --------------------------------------------------------------------


class StubRetriever:
    """Returns a fixed id list per query, and records what it was asked."""

    def __init__(self, results: dict[str, list[str]] | None = None) -> None:
        self.results = results or {}
        self.queries: list[str] = []

    def search(self, query: str, *, top_k: int = 20) -> list[str]:
        self.queries.append(query)
        return self.results.get(query, ["c1", "c2", "c3"])


class BrokenRetriever:
    def search(self, query: str, *, top_k: int = 20) -> list[str]:
        raise RetrievalUnavailable("chroma is down")


class StubStore:
    """Rehydrates chunk ids into Chunks, preserving requested order."""

    def fetch(self, chunk_ids: list[str]) -> list[Chunk]:
        return [
            Chunk(chunk_id=cid, source="cdc-htn", text=f"passage {cid}", ordinal=i)
            for i, cid in enumerate(chunk_ids)
        ]


class PassThroughReranker:
    def rerank(self, query: str, candidates: list[Chunk], *, top_n: int) -> list[Chunk]:
        return list(candidates[:top_n])


class ScriptedModel:
    """Plays a fixed script of responses, one per call.

    Each entry is what `draft` or `revise` should produce: (answer, [chunk_ids]).
    """

    def __init__(self, script: list[tuple[str, list[str]]]) -> None:
        self.script = list(script)
        self.calls = 0

    def __call__(self, prompt: str, chunks: list[Chunk]) -> tuple[str, list[Citation]]:
        answer, cited = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return answer, [
            Citation(chunk_id=cid, source="cdc-htn", quote="...") for cid in cited
        ]


def _planner(sub_questions: list[str]):
    def plan(query: str) -> list[str]:
        return sub_questions or [query]

    return plan


def _build(**overrides: Any):
    """Assemble the subgraph with sensible defaults; override what a test cares about."""
    from mesh.agents.guideline import build_guideline_subgraph

    defaults: dict[str, Any] = {
        "plan": _planner([]),
        "retriever": StubRetriever(),
        "store": StubStore(),
        "reranker": PassThroughReranker(),
        "draft": ScriptedModel([("Thiazides are first-line.", ["c1"])]),
        "detect_contradictions": lambda answer, chunks: None,
    }
    return build_guideline_subgraph(**{**defaults, **overrides})


def _run(subgraph, query: str = "what is first-line therapy for hypertension?"):
    return subgraph.invoke({"query": query})


# --- 1. planning ---------------------------------------------------------------


def test_a_simple_query_produces_one_sub_question():
    """Do not invent complexity: a single-part question is one retrieval."""
    subgraph = _build(plan=_planner(["what is first-line therapy for hypertension?"]))

    result = _run(subgraph)

    assert result["sub_questions"] == ["what is first-line therapy for hypertension?"]


def test_each_sub_question_is_retrieved_separately():
    """A multi-part question retrieves badly as one string."""
    retriever = StubRetriever({"first-line therapy?": ["c1"], "renal dosing?": ["c9"]})
    subgraph = _build(
        plan=_planner(["first-line therapy?", "renal dosing?"]), retriever=retriever
    )

    _run(subgraph)

    assert retriever.queries == ["first-line therapy?", "renal dosing?"]


def test_results_from_all_sub_questions_are_merged_without_duplicates():
    retriever = StubRetriever({"a": ["c1", "c2"], "b": ["c2", "c3"]})
    subgraph = _build(plan=_planner(["a", "b"]), retriever=retriever)

    result = _run(subgraph)

    assert sorted(result["retrieved_ids"]) == ["c1", "c2", "c3"]


# --- 2. retrieval failure ------------------------------------------------------


def test_an_unreachable_retriever_propagates_rather_than_being_swallowed():
    """The system refuses instead of answering from a degraded corpus. Catching
    this here would silently break that guarantee two layers down."""
    subgraph = _build(retriever=BrokenRetriever())

    with pytest.raises(RetrievalUnavailable):
        _run(subgraph)


# --- 3. reranking --------------------------------------------------------------


def test_only_the_top_chunks_reach_the_model():
    """Handing the model 20 mixed chunks instead of 5 strong ones costs money and
    dilutes the answer."""
    seen: list[list[Chunk]] = []

    def recording_draft(prompt: str, chunks: list[Chunk]) -> tuple[str, list[Citation]]:
        seen.append(chunks)
        return "answer", [Citation(chunk_id=chunks[0].chunk_id, source="s", quote="q")]

    retriever = StubRetriever({"q": [f"c{i}" for i in range(20)]})
    subgraph = _build(plan=_planner(["q"]), retriever=retriever, draft=recording_draft)

    _run(subgraph)

    assert len(seen[0]) == 5


# --- 4. citation verification --------------------------------------------------


def test_a_properly_cited_answer_is_returned_unchanged():
    subgraph = _build(draft=ScriptedModel([("Thiazides are first-line.", ["c1"])]))

    result = _run(subgraph)

    assert result["answer"] == "Thiazides are first-line."
    assert [c.chunk_id for c in result["citations"]] == ["c1"]


def test_an_answer_citing_an_unretrieved_chunk_triggers_a_revision():
    """The model invented a chunk id; ask it again before giving up."""
    model = ScriptedModel(
        [
            ("Thiazides are first-line.", ["ghost"]),
            ("Thiazides are first-line.", ["c1"]),
        ]
    )
    subgraph = _build(draft=model)

    result = _run(subgraph)

    assert model.calls == 2
    assert [c.chunk_id for c in result["citations"]] == ["c1"]
    assert result["revision_count"] == 1


def test_revision_stops_after_two_attempts_and_refuses():
    """Bounded on purpose. An unbounded loop is how one query costs $40.

    Note it refuses rather than returning the best attempt so far: every attempt
    cited something that was never retrieved, so there is no grounded answer to
    return.
    """
    from mesh.guardrails.nodes import UNGROUNDED_REFUSAL

    model = ScriptedModel([("Invented.", ["ghost"])])  # always ungrounded
    subgraph = _build(draft=model)

    result = _run(subgraph)

    assert result["revision_count"] == 2
    assert model.calls == 3  # initial draft plus two revisions
    assert result["answer"] == UNGROUNDED_REFUSAL
    assert result["citations"] == []


# --- 5. contradiction handling -------------------------------------------------


def test_disagreement_between_sources_is_surfaced_not_hidden():
    """Silently picking one source is the failure mode. Say they disagree."""
    subgraph = _build(
        detect_contradictions=lambda answer, chunks: "CDC and WHO differ on the target."
    )

    result = _run(subgraph)

    assert "differ" in result["answer"]
    assert "Thiazides are first-line." in result["answer"]


def test_agreeing_sources_add_no_note():
    subgraph = _build(detect_contradictions=lambda answer, chunks: None)

    result = _run(subgraph)

    assert result["answer"] == "Thiazides are first-line."


# --- 6. the parent contract ----------------------------------------------------


def test_the_subgraph_returns_what_the_mesh_needs():
    subgraph = _build()

    result = _run(subgraph)

    assert {"answer", "citations", "retrieved_ids"} <= set(result)


def test_retrieved_ids_are_reported_so_guard_out_can_verify_them():
    """guard_out checks citations against these. If the subgraph forgot to report
    them, every answer would be rejected as ungrounded by the parent."""
    subgraph = _build()

    result = _run(subgraph)

    assert set(c.chunk_id for c in result["citations"]) <= set(result["retrieved_ids"])
