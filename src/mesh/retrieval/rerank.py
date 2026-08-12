"""Cross-encoder reranking of fused retrieval results.

A bi-encoder (what the vector store uses) embeds query and document separately, so
it never compares them directly. A cross-encoder reads the pair together and is
markedly better at judging relevance — at the cost of one forward pass per
candidate, which is why it runs over ~20 fused candidates rather than the corpus.

The model runs locally: reranking 20 candidates per query through an API would
dominate the per-query bill, and a small cross-encoder on CPU is fast enough.

`sentence_transformers` is imported lazily inside the factory so this module — and
everything that imports it — works without the optional `rerank` extra installed.
"""

from collections.abc import Callable, Sequence

from mesh.retrieval.chunking import Chunk

# (query, passage) pairs in, relevance scores out.
PairScorer = Callable[[list[tuple[str, str]]], list[float]]

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self, *, score_pairs: PairScorer) -> None:
        self._score_pairs = score_pairs

    def rerank(self, query: str, candidates: Sequence[Chunk], *, top_n: int) -> list[Chunk]:
        """Reorder candidates by cross-encoder relevance and keep the best `top_n`.

        On scorer failure the incoming order is returned instead of raising. That is
        the opposite of the retrieval contract, and deliberately so: reranking only
        reorders evidence that is already retrieved and grounded, so losing it costs
        answer quality. Losing retrieval changes what evidence exists at all.
        """
        if not candidates:
            return []

        pairs = [(query, chunk.text) for chunk in candidates]
        try:
            scores = self._score_pairs(pairs)
        except Exception:  # noqa: BLE001 - reranking is an enhancement, not a gate
            return list(candidates[:top_n])

        # Negate the score rather than reversing, so Python's stable sort preserves
        # the incoming fusion order among ties.
        ordered = sorted(zip(candidates, scores, strict=True), key=lambda pair: -pair[1])

        return [chunk for chunk, _ in ordered[:top_n]]


def build_cross_encoder(model_name: str = DEFAULT_MODEL) -> PairScorer:
    """Load a local cross-encoder. Requires the `rerank` extra."""
    from sentence_transformers import CrossEncoder  # noqa: PLC0415 - optional dependency

    model = CrossEncoder(model_name)

    def score_pairs(pairs: list[tuple[str, str]]) -> list[float]:
        return [float(score) for score in model.predict(pairs)]

    return score_pairs
