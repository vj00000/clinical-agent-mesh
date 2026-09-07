"""Ingestion orchestration: documents in, embedded chunks in the store.

The store and embedder are small recording stubs rather than mocks, so these
assert on what actually reached the store.
"""

from collections.abc import Sequence

from mesh.retrieval.chunking import Chunk
from mesh.retrieval.documents import Document
from mesh.retrieval.ingest import (
    COLLECTION_BY_ROUTE,
    CORPORA,
    dedupe_documents,
    ingest_documents,
)


class RecordingStore:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.vectors: list[Sequence[float]] = []

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        self.chunks.extend(chunks)
        self.vectors.extend(vectors)


class CountingEmbedder:
    """Returns a one-dimensional vector per text and counts how many it saw."""

    def __init__(self) -> None:
        self.calls = 0
        self.texts_embedded = 0

    def __call__(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        self.texts_embedded += len(texts)
        return [[float(len(text))] for text in texts]


def _doc(doc_id: str, words: int) -> Document:
    return Document(
        doc_id=doc_id,
        source="pubmed",
        title="t",
        text=" ".join(f"w{i}" for i in range(words)),
    )


def test_each_document_is_chunked_and_stored():
    store, embedder = RecordingStore(), CountingEmbedder()

    report = ingest_documents(
        [_doc("pmid:1", 250)], store=store, embed=embedder, target_words=100, overlap_words=20
    )

    assert report.documents == 1
    assert report.chunks == len(store.chunks) > 1


def test_every_stored_chunk_has_a_matching_vector():
    store, embedder = RecordingStore(), CountingEmbedder()

    ingest_documents(
        [_doc("pmid:1", 250)], store=store, embed=embedder, target_words=100, overlap_words=20
    )

    assert len(store.vectors) == len(store.chunks)


def test_chunks_are_embedded_in_one_batched_call_per_document():
    """Per-chunk embedding calls would multiply latency and cost for no benefit."""
    embedder = CountingEmbedder()
    recording = RecordingStore()

    ingest_documents(
        [_doc("pmid:1", 250), _doc("pmid:2", 250)],
        store=recording,
        embed=embedder,
        target_words=100,
        overlap_words=20,
    )

    assert embedder.calls == 2
    assert embedder.texts_embedded == len(recording.chunks)


def test_chunks_carry_the_document_id_as_their_source():
    store, embedder = RecordingStore(), CountingEmbedder()

    ingest_documents([_doc("pmid:1", 50)], store=store, embed=embedder, target_words=100)

    assert {c.source for c in store.chunks} == {"pmid:1"}


def test_a_document_with_no_usable_text_is_reported_as_skipped():
    store, embedder = RecordingStore(), CountingEmbedder()
    blank = Document(doc_id="pmid:9", source="pubmed", title="t", text="   ")

    report = ingest_documents([blank], store=store, embed=embedder)

    assert report.documents == 0
    assert report.skipped == 1
    assert store.chunks == []


def test_documents_repeated_across_queries_are_deduplicated_by_id():
    """The same MedlinePlus topic surfaces under several queries; embedding it
    twice costs money for an identical result."""
    docs = [_doc("pmid:1", 10), _doc("pmid:2", 10), _doc("pmid:1", 10)]

    assert [d.doc_id for d in dedupe_documents(docs)] == ["pmid:1", "pmid:2"]


def test_deduplication_keeps_the_first_occurrence():
    first = Document(doc_id="pmid:1", source="pubmed", title="first", text="a")
    later = Document(doc_id="pmid:1", source="pubmed", title="second", text="b")

    assert dedupe_documents([first, later])[0].title == "first"


def test_ingesting_nothing_does_not_call_the_embedder():
    store, embedder = RecordingStore(), CountingEmbedder()

    report = ingest_documents([], store=store, embed=embedder)

    assert report.chunks == 0
    assert embedder.calls == 0


def test_every_specialist_route_maps_to_a_collection():
    """Reaches for the private name on purpose. This is the contract between the
    mesh's routes and the corpora, and adding a fifth specialist without giving
    it something to retrieve from should break loudly here.
    """
    from mesh.graph import _SPECIALIST_NAMES

    assert set(COLLECTION_BY_ROUTE) == set(_SPECIALIST_NAMES)


def test_each_corpus_fills_a_distinct_collection():
    collections = [corpus.collection for corpus in CORPORA]

    assert len(collections) == len(set(collections))


def test_no_corpus_fills_a_collection_nothing_reads():
    """Embedding text no agent will ever retrieve is money spent for nothing."""
    assert {corpus.collection for corpus in CORPORA} <= set(COLLECTION_BY_ROUTE.values())


def test_coverage_is_the_only_collection_still_unbuilt():
    """The Medicare Coverage Database publishes bulk downloads rather than a
    queryable API, so `coverage` cannot follow the fetch-on-demand pattern the
    other three use. Named here so the gap cannot be quietly forgotten: when the
    CMS corpus lands this test fails, and gets deleted.
    """
    built = {corpus.collection for corpus in CORPORA}

    assert set(COLLECTION_BY_ROUTE.values()) - built == {"coverage"}
