"""The discharge specialist subgraph.

Interactions come from a tool, not from recall; the reading grade is computed, not
judged; and the warnings are appended verbatim. These tests hold those lines.
"""

from typing import Any

import pytest

from mesh.agents.discharge import (
    INTERACTION_HEADING,
    Interaction,
    Medication,
    build_discharge_subgraph,
)
from mesh.retrieval.chunking import Chunk
from mesh.retrieval.hybrid import RetrievalUnavailable
from mesh.state import Citation

QUERY = "what do I need to know about the tablets I was sent home with?"

PLAIN = "Take one pill each day. Drink water with it."

CLINICAL = (
    "Discontinue the anticoagulant medication immediately if unexplained "
    "haemorrhage manifests and consult your prescribing physician regarding "
    "alternative antithrombotic therapy."
)

WARFARIN = Medication(name="warfarin", dose="5mg", frequency="daily")
IBUPROFEN = Medication(name="ibuprofen", dose="400mg", frequency="as needed")

BLEEDING_RISK = Interaction(
    drugs=["warfarin", "ibuprofen"],
    warning="together these raise the risk of bleeding",
    chunk_id="openfda-0093",
    source="openfda",
)


class StubRetriever:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query: str, *, top_k: int = 20) -> list[str]:
        self.queries.append(query)
        return ["c0", "c1"]


class BrokenRetriever:
    def search(self, query: str, *, top_k: int = 20) -> list[str]:
        raise RetrievalUnavailable("chroma is down")


class StubStore:
    def fetch(self, chunk_ids: list[str]) -> list[Chunk]:
        return [
            Chunk(chunk_id=cid, source="openfda-label", text=f"label {cid}", ordinal=i)
            for i, cid in enumerate(chunk_ids)
        ]


class PassThroughReranker:
    def rerank(self, query: str, candidates: list[Chunk], *, top_n: int) -> list[Chunk]:
        return list(candidates[:top_n])


class ScriptedDrafter:
    """Plays a fixed script, one entry per call, repeating the last."""

    def __init__(self, *answers: str) -> None:
        self.answers = answers
        self.calls = 0

    def __call__(self, prompt: str, chunks: list[Chunk]) -> tuple[str, list[Citation]]:
        answer = self.answers[min(self.calls, len(self.answers) - 1)]
        self.calls += 1
        return answer, [Citation(chunk_id="c0", source="openfda-label", quote="...")]


class RecordingLookup:
    def __init__(self, *interactions: Interaction) -> None:
        self.interactions = interactions
        self.names: list[list[str]] = []

    def __call__(self, names: list[str]) -> list[Interaction]:
        self.names.append(names)
        return list(self.interactions)


def _build(**overrides: Any):
    defaults: dict[str, Any] = {
        "extract_medications": lambda query: [WARFARIN, IBUPROFEN],
        "lookup_interactions": RecordingLookup(BLEEDING_RISK),
        "retriever": StubRetriever(),
        "store": StubStore(),
        "reranker": PassThroughReranker(),
        "draft": ScriptedDrafter(PLAIN),
    }
    return build_discharge_subgraph(**{**defaults, **overrides})


def test_the_interaction_tool_receives_the_extracted_medication_names():
    lookup = RecordingLookup(BLEEDING_RISK)

    _build(lookup_interactions=lookup).invoke({"query": QUERY})

    assert lookup.names == [["warfarin", "ibuprofen"]]


def test_a_single_medication_never_calls_the_interaction_tool():
    """One drug cannot interact with anything. A network round trip guaranteed to
    return nothing is still a network round trip."""
    lookup = RecordingLookup()

    _build(
        extract_medications=lambda query: [WARFARIN],
        lookup_interactions=lookup,
    ).invoke({"query": QUERY})

    assert lookup.names == []


def test_retrieval_searches_the_drug_names_not_the_question():
    """Label text is indexed under the drug, not under how the patient asked."""
    retriever = StubRetriever()

    _build(retriever=retriever).invoke({"query": QUERY})

    assert retriever.queries == ["warfarin ibuprofen"]


def test_interactions_are_appended_verbatim_and_cited():
    result = _build().invoke({"query": QUERY})

    assert INTERACTION_HEADING in result["answer"]
    assert "together these raise the risk of bleeding" in result["answer"]
    assert "openfda-0093" in [citation.chunk_id for citation in result["citations"]]


def test_interaction_ids_join_retrieved_ids_so_guard_out_accepts_them():
    """A tool result is evidence. Without its id in retrieved_ids, guard_out sees
    a citation for a chunk that was never retrieved and refuses the answer."""
    result = _build().invoke({"query": QUERY})

    cited = {citation.chunk_id for citation in result["citations"]}
    assert cited <= set(result["retrieved_ids"])


def test_no_interactions_leaves_the_instructions_untouched():
    result = _build(lookup_interactions=RecordingLookup()).invoke({"query": QUERY})

    assert result["answer"] == PLAIN


def test_hard_to_read_instructions_are_rewritten_once():
    drafter = ScriptedDrafter(CLINICAL, PLAIN)

    result = _build(draft=drafter).invoke({"query": QUERY})

    assert drafter.calls == 2
    assert result["reading_grade"] < 8.0


def test_rewriting_is_bounded_to_one_pass():
    """A second rewrite of already-simplified text drops clinical detail rather
    than shortening sentences, so the hard version is accepted instead."""
    drafter = ScriptedDrafter(CLINICAL)

    result = _build(draft=drafter).invoke({"query": QUERY})

    assert drafter.calls == 2
    assert result["simplifications"] == 1
    assert result["reading_grade"] > 8.0


def test_readable_instructions_are_not_rewritten():
    drafter = ScriptedDrafter(PLAIN)

    _build(draft=drafter).invoke({"query": QUERY})

    assert drafter.calls == 1


def test_an_unreachable_retriever_propagates():
    subgraph = _build(retriever=BrokenRetriever())

    with pytest.raises(RetrievalUnavailable):
        subgraph.invoke({"query": QUERY})


def test_the_subgraph_returns_what_the_mesh_needs():
    result = _build().invoke({"query": QUERY})

    assert {"answer", "citations", "retrieved_ids"} <= set(result)
