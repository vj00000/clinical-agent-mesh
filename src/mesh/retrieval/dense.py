"""Dense retrieval backed by Chroma.

The embedding function is injected rather than configured inside Chroma, so the
same vectors are used at ingest and query time and tests can run without an API
key.
"""

import contextlib
from collections.abc import Callable, Sequence

import chromadb

from mesh.retrieval.chunking import Chunk

Embedder = Callable[[str], list[float]]


class ChromaDense:
    """Semantic half of hybrid retrieval."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        collection: str,
        embed_query: Embedder,
    ) -> None:
        self._client = chromadb.HttpClient(host=host, port=port)
        self._collection_name = collection
        self._embed_query = embed_query

    @property
    def _collection(self) -> chromadb.Collection:
        return self._client.get_or_create_collection(
            self._collection_name, metadata={"hnsw:space": "cosine"}
        )

    def reset(self) -> None:
        """Drop the collection. Used by tests and by a full re-ingest."""
        # An absent collection is the desired end state either way.
        with contextlib.suppress(Exception):
            self._client.delete_collection(self._collection_name)

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        """Store chunks by content-addressed id, so re-ingest overwrites rather than duplicates."""
        if not chunks:
            return

        embeddings: list[Sequence[float]] = [list(v) for v in vectors]
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source, "ordinal": c.ordinal} for c in chunks],
        )

    def search(self, query: str, *, top_k: int = 20) -> list[str]:
        collection = self._collection
        if collection.count() == 0:
            return []

        query_embeddings: list[Sequence[float]] = [self._embed_query(query)]
        result = collection.query(
            query_embeddings=query_embeddings,
            n_results=min(top_k, collection.count()),
        )
        ids = result.get("ids") or [[]]
        return list(ids[0])
