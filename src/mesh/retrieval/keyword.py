"""BM25 keyword retrieval over chunks."""

import re
from collections.abc import Sequence

from rank_bm25 import BM25Okapi

from mesh.retrieval.chunking import Chunk

_TOKEN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Index:
    """Lexical half of hybrid retrieval."""

    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self._chunk_ids = [c.chunk_id for c in chunks]
        self._bm25 = BM25Okapi([tokenize(c.text) for c in chunks]) if chunks else None

    def search(self, query: str, *, top_k: int = 20) -> list[str]:
        """Return chunk ids ranked by lexical relevance, best first.

        Zero-scoring chunks are dropped: a chunk sharing no terms with the query
        is noise, and passing it to fusion would dilute the dense results.
        """
        if self._bm25 is None:
            return []

        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(
            (
                (chunk_id, score)
                for chunk_id, score in zip(self._chunk_ids, scores, strict=True)
                if score > 0
            ),
            key=lambda pair: -pair[1],
        )
        return [chunk_id for chunk_id, _ in ranked[:top_k]]
