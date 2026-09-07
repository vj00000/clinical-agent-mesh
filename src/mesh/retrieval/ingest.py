"""Ingestion: source documents in, embedded chunks in the vector store.

The embedder and store are injected so ingestion is testable without an API key
and so the same code path serves any corpus.
"""

import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import httpx
from pydantic import BaseModel, ValidationError

from mesh.models.config import Settings
from mesh.models.providers import build_embeddings
from mesh.retrieval.chunking import Chunk, chunk_document
from mesh.retrieval.dense import ChromaDense
from mesh.retrieval.documents import Document
from mesh.retrieval.sources import fetch_medlineplus, fetch_openfda_labels, fetch_pubmed

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

# Symptoms the triage specialist is expected to field. MedlinePlus only, and
# deliberately: triage answers in the register a worried person reads, while
# PubMed abstracts are written for clinicians.
TRIAGE_SYMPTOMS = (
    "chest pain",
    "shortness of breath",
    "headache",
    "fever",
    "abdominal pain",
    "dizziness",
)

# Medications that turn up most often in discharge summaries, chosen so the
# interaction lookup has real pairs to find.
DISCHARGE_DRUGS = (
    "warfarin",
    "metformin",
    "lisinopril",
    "atorvastatin",
    "amoxicillin",
    "ibuprofen",
)

# Which collection each specialist retrieves from. The names differ on purpose:
# a collection is named for what it holds, an agent for what it does. The
# composition root reads this to build each agent its own retriever.
COLLECTION_BY_ROUTE = {
    "guideline": "guideline",
    "triage": "triage",
    "prior_auth": "coverage",
    "discharge": "drug",
}

CorpusBuilder = Callable[[httpx.Client], list[Document]]


@dataclass(frozen=True)
class Corpus:
    """One collection and the function that fills it."""

    collection: str
    build: CorpusBuilder


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


def build_triage_corpus(client: httpx.Client, *, per_symptom: int = 5) -> list[Document]:
    """Fetch patient-facing symptom pages for the triage specialist."""
    documents: list[Document] = []

    for symptom in TRIAGE_SYMPTOMS:
        print(f"  fetching {symptom}...", flush=True)
        documents.extend(fetch_medlineplus(symptom, limit=per_symptom, client=client))

    return dedupe_documents(documents)


def build_drug_corpus(client: httpx.Client, *, per_drug: int = 3) -> list[Document]:
    """Fetch openFDA labels for the medications discharge summaries mention most."""
    documents: list[Document] = []

    for drug in DISCHARGE_DRUGS:
        print(f"  fetching {drug}...", flush=True)
        documents.extend(fetch_openfda_labels(drug, limit=per_drug, client=client))

    return dedupe_documents(documents)


# `coverage` is absent: the Medicare Coverage Database publishes bulk downloads
# rather than a queryable API, so prior_auth cannot follow the fetch-on-demand
# pattern the other three use. Until it lands, that route retrieves nothing.
CORPORA = (
    Corpus(collection=COLLECTION_BY_ROUTE["guideline"], build=build_guideline_corpus),
    Corpus(collection=COLLECTION_BY_ROUTE["triage"], build=build_triage_corpus),
    Corpus(collection=COLLECTION_BY_ROUTE["discharge"], build=build_drug_corpus),
)


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

    with httpx.Client(timeout=30.0) as client:
        for corpus in CORPORA:
            print(f"\n[{corpus.collection}] fetching...")
            documents = corpus.build(client)
            print(f"  {len(documents)} unique documents")

            store = ChromaDense(
                host=settings.chroma_host,
                port=settings.chroma_port,
                collection=corpus.collection,
                embed_query=embeddings.embed_query,
            )
            report = ingest_documents(documents, store=store, embed=embeddings.embed_documents)

            print(
                f"  ingested {report.documents} documents into {report.chunks} chunks "
                f"({report.skipped} skipped, no usable text)"
            )

    # Said out loud rather than left to be discovered when a route answers
    # "insufficient evidence" to everything.
    missing = sorted(set(COLLECTION_BY_ROUTE.values()) - {c.collection for c in CORPORA})
    if missing:
        print(f"\nNot built: {', '.join(missing)}. Routes reading them find an empty index.")


if __name__ == "__main__":
    main()
