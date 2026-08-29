"""The prior-auth specialist subgraph.

The model marks criteria against the retrieved policy; the verdict and the words
for it are computed. These tests check the wiring holds that line.
"""

from typing import Any

import pytest

from mesh.agents.prior_auth import build_prior_auth_subgraph
from mesh.agents.prior_auth_rules import Criterion, Decision
from mesh.retrieval.chunking import Chunk
from mesh.retrieval.hybrid import RetrievalUnavailable

QUERY = "is continuous glucose monitoring covered for this patient?"


class StubRetriever:
    def search(self, query: str, *, top_k: int = 20) -> list[str]:
        return [f"c{i}" for i in range(20)]


class BrokenRetriever:
    def search(self, query: str, *, top_k: int = 20) -> list[str]:
        raise RetrievalUnavailable("chroma is down")


class StubStore:
    def fetch(self, chunk_ids: list[str]) -> list[Chunk]:
        return [
            Chunk(chunk_id=cid, source="cms-lcd", text=f"policy {cid}", ordinal=i)
            for i, cid in enumerate(chunk_ids)
        ]


class PassThroughReranker:
    def rerank(self, query: str, candidates: list[Chunk], *, top_n: int) -> list[Chunk]:
        return list(candidates[:top_n])


def _reader(*criteria: Criterion):
    def read(query: str, chunks: list[Chunk]) -> list[Criterion]:
        return list(criteria)

    return read


def _build(**overrides: Any):
    defaults: dict[str, Any] = {
        "retriever": StubRetriever(),
        "store": StubStore(),
        "reranker": PassThroughReranker(),
        "read_criteria": _reader(
            Criterion(text="documented trial of metformin", met=True, chunk_id="c0")
        ),
    }
    return build_prior_auth_subgraph(**{**defaults, **overrides})


def test_an_approval_cites_the_policy_it_rests_on():
    result = _build().invoke({"query": QUERY})

    assert result["decision"] is Decision.APPROVE
    assert [c.chunk_id for c in result["citations"]] == ["c0"]
    assert result["citations"][0].source == "cms-lcd"


def test_a_denial_is_computed_not_written():
    """The reader marks a criterion unmet. Nothing in a prompt says "deny"."""
    subgraph = _build(
        read_criteria=_reader(Criterion(text="HbA1c above 7.5", met=False, chunk_id="c0"))
    )

    result = subgraph.invoke({"query": QUERY})

    assert result["decision"] is Decision.DENY
    assert "not met" in result["answer"]


def test_an_unresolved_criterion_is_routed_as_a_question():
    """more_info asks rather than asserts, so guard_out must not hold it to the
    standard of a grounded clinical claim."""
    subgraph = _build(
        read_criteria=_reader(
            Criterion(text="documented trial of metformin", met=None, chunk_id="c0")
        )
    )

    result = subgraph.invoke({"query": QUERY})

    assert result["decision"] is Decision.MORE_INFO
    assert result["route"] == "clarify"


def test_a_criterion_citing_an_unretrieved_chunk_is_kept_not_dropped():
    """Dropping it would remove the criterion from the checklist, and dropping a
    failed criterion flips a denial into an approval. It is kept, cited with a
    placeholder source, and guard_out rejects the answer downstream."""
    subgraph = _build(
        read_criteria=_reader(Criterion(text="invented requirement", met=False, chunk_id="ghost"))
    )

    result = subgraph.invoke({"query": QUERY})

    assert result["decision"] is Decision.DENY
    assert [c.chunk_id for c in result["citations"]] == ["ghost"]


def test_only_the_top_chunks_reach_the_reader():
    seen: list[list[Chunk]] = []

    def recording_reader(query: str, chunks: list[Chunk]) -> list[Criterion]:
        seen.append(chunks)
        return [Criterion(text="a requirement", met=True, chunk_id=chunks[0].chunk_id)]

    _build(read_criteria=recording_reader).invoke({"query": QUERY})

    assert len(seen[0]) == 5


def test_an_unreachable_retriever_propagates():
    subgraph = _build(retriever=BrokenRetriever())

    with pytest.raises(RetrievalUnavailable):
        subgraph.invoke({"query": QUERY})


def test_the_subgraph_returns_what_the_mesh_needs():
    result = _build().invoke({"query": QUERY})

    assert {"answer", "citations", "retrieved_ids"} <= set(result)
