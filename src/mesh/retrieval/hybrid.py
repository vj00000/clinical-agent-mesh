"""Hybrid retrieval: dense semantic search fused with BM25 lexical search."""

from typing import Protocol

from mesh.retrieval.fusion import reciprocal_rank_fusion
from mesh.retrieval.keyword import BM25Index

# Each half is over-fetched before fusion so a chunk ranked mid-list by both
# retrievers can still win, which is the entire point of RRF.
_CANDIDATES_PER_RETRIEVER = 20


class RetrievalUnavailable(RuntimeError):
    """The vector backend could not be reached.

    Raised rather than falling back to keyword-only results: answering from a
    degraded corpus is how a grounded system quietly starts hallucinating.
    """


class DenseRetriever(Protocol):
    def search(self, query: str, *, top_k: int) -> list[str]: ...


class HybridRetriever:
    def __init__(self, *, dense: DenseRetriever, keyword: BM25Index) -> None:
        self._dense = dense
        self._keyword = keyword

    def search(self, query: str, *, top_k: int = 20) -> list[str]:
        try:
            dense_hits = self._dense.search(query, top_k=_CANDIDATES_PER_RETRIEVER)
        except Exception as exc:  # noqa: BLE001 - any backend failure is a refusal
            raise RetrievalUnavailable(str(exc)) from exc

        keyword_hits = self._keyword.search(query, top_k=_CANDIDATES_PER_RETRIEVER)

        return reciprocal_rank_fusion([dense_hits, keyword_hits])[:top_k]
