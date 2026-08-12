"""Ingestion: source documents in, embedded chunks in the vector store.

The embedder and store are injected so ingestion is testable without an API key
and so the same code path serves any corpus.
"""

import sys
from collections.abc import Callable, Sequence
from typing import Protocol

import httpx
from pydantic import BaseModel, ValidationError

from mesh.models.config import Settings
from mesh.models.providers import build_embeddings
from mesh.retrieval.chunking import Chunk, chunk_document
from mesh.retrieval.dense import ChromaDense
from mesh.retrieval.documents import Document
from mesh.retrieval.sources import fetch_medlineplus, fetch_pubmed

# Takes list, not Sequence: LangChain's embed_documents is typed for list[str],
# and callable parameters are contravariant, so a Sequence-typed alias rejects it.
BatchEmbedder = Callable[[list[str]], list[list[float]]]

# Conditions the guideline copilot is expected to answer on. Kept small on
# purpose: a focused corpus retrieves better than a broad shallow one, and the
# eval golden set has to be hand-written against whatever is ingested here.
GUIDELINE_TOPICS = (
    "hypertension",
    "type 2 diabetes",
    "asthma",
    "heart failure",
    "atrial fibrillation",
    "chronic kidney disease",
)

GUIDELINE_COLLECTION = "guideline"


class ChunkStore(Protocol):
    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None: ...


class IngestReport(BaseModel):
    documents: int = 0
    chunks: int = 0
    skipped: int = 0


def dedupe_documents(documents: Sequence[Document]) -> list[Document]:
    """Drop repeated doc_ids, keeping the first occurrence.

    Topic queries overlap, so the same source document arrives more than once.
    Embedding it twice costs money for an identical result.
    """
    seen: set[str] = set()
    unique: list[Document] = []
    for document in documents:
        if document.doc_id in seen:
            continue
        seen.add(document.doc_id)
        unique.append(document)

    return unique


def ingest_documents(
    documents: Sequence[Document],
    *,
    store: ChunkStore,
    embed: BatchEmbedder,
    target_words: int = 300,
    overlap_words: int = 50,
) -> IngestReport:
    """Chunk, embed, and upsert each document.

    Embedding is batched per document: one call per chunk would multiply latency
    and cost for no benefit. Upserts are keyed by content-addressed chunk id, so
    re-running this overwrites rather than duplicating.
    """
    report = IngestReport()

    for document in documents:
        chunks = chunk_document(
            document.text,
            source=document.doc_id,
            target_words=target_words,
            overlap_words=overlap_words,
        )
        if not chunks:
            report.skipped += 1
            continue

        vectors = embed([c.text for c in chunks])
        store.upsert(chunks, vectors)

        report.documents += 1
        report.chunks += len(chunks)

    return report


def build_guideline_corpus(client: httpx.Client, *, per_topic: int = 5) -> list[Document]:
    """Fetch the guideline corpus: patient-facing topics plus clinical abstracts."""
    documents: list[Document] = []

    for topic in GUIDELINE_TOPICS:
        print(f"  fetching {topic}...", flush=True)
        documents.extend(fetch_medlineplus(topic, limit=per_topic, client=client))
        documents.extend(fetch_pubmed(f"{topic} guideline", limit=per_topic, client=client))

    return dedupe_documents(documents)


def main() -> None:
    """Entry point for `make ingest`."""
    try:
        settings = Settings()
    except ValidationError:
        print(
            "OPENAI_API_KEY is not set. Run `cp .env.example .env` and add your key.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None

    embeddings = build_embeddings(settings)
    store = ChromaDense(
        host=settings.chroma_host,
        port=settings.chroma_port,
        collection=GUIDELINE_COLLECTION,
        embed_query=embeddings.embed_query,
    )

    print(f"Fetching corpus for {len(GUIDELINE_TOPICS)} topics...")
    with httpx.Client(timeout=30.0) as client:
        documents = build_guideline_corpus(client)
    print(f"Fetched {len(documents)} unique documents.")

    print("Chunking and embedding...")
    report = ingest_documents(documents, store=store, embed=embeddings.embed_documents)

    print(
        f"Ingested {report.documents} documents into {report.chunks} chunks "
        f"({report.skipped} skipped, no usable text)."
    )


if __name__ == "__main__":
    main()
