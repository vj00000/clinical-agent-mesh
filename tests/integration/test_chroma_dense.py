"""Dense retrieval against a real Chroma instance.

The embedder is injected, so these run against real Chroma without an API key
and without pretending a mocked vector store proves anything.
"""

import httpx
import pytest

from mesh.retrieval.chunking import Chunk
from mesh.retrieval.dense import ChromaDense

CHROMA_URL = "http://localhost:8001"

TOPIC_WORDS = ("hypertension", "diabetes", "cholesterol")


def toy_embed(text: str) -> list[float]:
    """Deterministic topic-count embedding — no API key, no randomness."""
    lowered = text.lower()
    return [float(lowered.count(word)) for word in TOPIC_WORDS]


def _chroma_is_up() -> bool:
    try:
        return httpx.get(f"{CHROMA_URL}/api/v2/heartbeat", timeout=2.0).status_code == 200
    except httpx.HTTPError:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _chroma_is_up(), reason="Chroma not running: `make up`"),
]


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, source="test", text=text, ordinal=0)


CORPUS = [
    _chunk("c1", "hypertension hypertension management guidance"),
    _chunk("c2", "diabetes diabetes therapy guidance"),
    _chunk("c3", "cholesterol cholesterol lowering guidance"),
]


@pytest.fixture
def store(request):
    # Chroma rejects a name that does not end in a letter or digit, and
    # truncating a long test name can land on an underscore.
    collection = f"test_{request.node.name}"[:60].rstrip("_.-")
    dense = ChromaDense(host="localhost", port=8001, collection=collection, embed_query=toy_embed)
    dense.reset()
    yield dense
    dense.reset()


def test_an_upserted_chunk_is_retrieved_by_its_topic(store):
    store.upsert(CORPUS, [toy_embed(c.text) for c in CORPUS])

    results = store.search("hypertension", top_k=3)

    assert results[0] == "c1"


def test_top_k_limits_the_number_of_dense_results(store):
    store.upsert(CORPUS, [toy_embed(c.text) for c in CORPUS])

    results = store.search("diabetes", top_k=2)

    assert len(results) == 2


def test_searching_an_empty_collection_returns_nothing(store):
    results = store.search("hypertension", top_k=3)

    assert results == []


def test_fetch_returns_the_chunks_for_the_given_ids(store):
    """Retrieval yields ids; reranking needs the text behind them."""
    store.upsert(CORPUS, [toy_embed(c.text) for c in CORPUS])

    chunks = store.fetch(["c2", "c1"])

    assert [c.chunk_id for c in chunks] == ["c2", "c1"]
    assert chunks[0].text == "diabetes diabetes therapy guidance"


def test_fetch_preserves_the_requested_order(store):
    """The requested order is the fused ranking; Chroma does not promise to keep it,
    so losing it would silently discard the ranking before reranking even runs."""
    store.upsert(CORPUS, [toy_embed(c.text) for c in CORPUS])

    chunks = store.fetch(["c3", "c1", "c2"])

    assert [c.chunk_id for c in chunks] == ["c3", "c1", "c2"]


def test_fetch_skips_ids_that_are_not_present(store):
    store.upsert(CORPUS, [toy_embed(c.text) for c in CORPUS])

    chunks = store.fetch(["c1", "missing"])

    assert [c.chunk_id for c in chunks] == ["c1"]


def test_fetching_nothing_returns_nothing(store):
    assert store.fetch([]) == []


def test_fetch_restores_the_source_and_ordinal(store):
    store.upsert(CORPUS, [toy_embed(c.text) for c in CORPUS])

    chunk = store.fetch(["c1"])[0]

    assert chunk.source == "test"
    assert chunk.ordinal == 0


def test_reupserting_the_same_chunk_does_not_duplicate_it(store):
    store.upsert(CORPUS, [toy_embed(c.text) for c in CORPUS])
    store.upsert(CORPUS, [toy_embed(c.text) for c in CORPUS])

    results = store.search("hypertension", top_k=10)

    assert len(results) == len(set(results)) == 3


def test_all_chunks_returns_the_whole_collection(store):
    """The lexical index is built from this. BM25 scores against the corpus in
    memory, so it cannot be assembled from search results."""
    store.upsert(CORPUS, [toy_embed(c.text) for c in CORPUS])

    chunks = store.all_chunks()

    assert {c.chunk_id for c in chunks} == {"c1", "c2", "c3"}


def test_all_chunks_of_an_empty_collection_is_empty(store):
    assert store.all_chunks() == []


def test_all_chunks_pages_through_a_collection_larger_than_one_batch(store):
    """A single get() of a large collection is a large response held twice."""
    store.upsert(CORPUS, [toy_embed(c.text) for c in CORPUS])

    chunks = store.all_chunks(batch_size=2)

    assert {c.chunk_id for c in chunks} == {"c1", "c2", "c3"}


def test_all_chunks_restores_the_text_so_bm25_can_tokenize_it(store):
    store.upsert(CORPUS, [toy_embed(c.text) for c in CORPUS])

    by_id = {c.chunk_id: c for c in store.all_chunks()}

    assert by_id["c2"].text == "diabetes diabetes therapy guidance"
