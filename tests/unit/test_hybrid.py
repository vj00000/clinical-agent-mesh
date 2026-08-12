"""Hybrid retrieval: dense + BM25 fused by rank.

The dense half is exercised through small real stubs rather than a mocking
library, so these assert on fusion behaviour instead of on call bookkeeping.
"""

import pytest

from mesh.retrieval.chunking import Chunk
from mesh.retrieval.hybrid import HybridRetriever, RetrievalUnavailable
from mesh.retrieval.keyword import BM25Index


class StubDense:
    def __init__(self, results: list[str]) -> None:
        self._results = results

    def search(self, query: str, *, top_k: int) -> list[str]:
        return self._results[:top_k]


class BrokenDense:
    def search(self, query: str, *, top_k: int) -> list[str]:
        raise ConnectionError("chroma refused the connection")


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, source="test", text=text, ordinal=0)


# Four documents, not two: BM25Okapi's IDF is log(N-df+0.5) - log(df+0.5), which
# is exactly 0 when N=2 and df=1, so every score collapses to zero on a two-doc
# corpus. That degeneracy is a small-corpus artifact, not a property of any real
# corpus, so the fixture stays above it.
CORPUS = [
    _chunk("c1", "Lisinopril is an ACE inhibitor for hypertension."),
    _chunk("c2", "Metformin is first-line therapy for type 2 diabetes."),
    _chunk("c3", "Blood pressure should be measured after five minutes of rest."),
    _chunk("c4", "Atorvastatin is prescribed to lower LDL cholesterol."),
]


def test_a_chunk_both_retrievers_agree_on_is_ranked_first():
    retriever = HybridRetriever(dense=StubDense(["c2", "c1"]), keyword=BM25Index(CORPUS))

    results = retriever.search("lisinopril", top_k=2)

    assert results[0] == "c1"


def test_dense_results_are_returned_when_the_keyword_index_matches_nothing():
    retriever = HybridRetriever(dense=StubDense(["c2"]), keyword=BM25Index(CORPUS))

    results = retriever.search("xyzzy", top_k=5)

    assert results == ["c2"]


def test_keyword_results_are_returned_when_dense_finds_nothing():
    retriever = HybridRetriever(dense=StubDense([]), keyword=BM25Index(CORPUS))

    results = retriever.search("metformin", top_k=5)

    assert results == ["c2"]


def test_top_k_limits_the_fused_results():
    retriever = HybridRetriever(dense=StubDense(["c1", "c2"]), keyword=BM25Index(CORPUS))

    results = retriever.search("hypertension diabetes", top_k=1)

    assert len(results) == 1


def test_an_unreachable_dense_backend_raises_rather_than_degrading_silently():
    """The graph must refuse, not answer from keyword hits alone."""
    retriever = HybridRetriever(dense=BrokenDense(), keyword=BM25Index(CORPUS))

    with pytest.raises(RetrievalUnavailable):
        retriever.search("lisinopril", top_k=5)
