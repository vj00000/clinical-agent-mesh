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

    def fetch(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        """Rehydrate chunks by id, in the order requested.

        Retrieval yields ids; reranking needs the text behind them. The requested
        order is the fused ranking and Chroma does not promise to preserve it, so
        the result is reordered here rather than trusted.
        """
        if not chunk_ids:
            return []

        result = self._collection.get(ids=list(chunk_ids))
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []

        by_id: dict[str, Chunk] = {}
        for chunk_id, text, metadata in zip(result["ids"], documents, metadatas, strict=True):
            meta = metadata or {}
            by_id[chunk_id] = Chunk(
                chunk_id=chunk_id,
                source=str(meta.get("source", "")),
                text=text or "",
                ordinal=int(meta.get("ordinal", 0)),  # type: ignore[arg-type]
            )

        return [by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in by_id]

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
