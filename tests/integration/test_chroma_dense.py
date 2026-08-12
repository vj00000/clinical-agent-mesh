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
    collection = f"test_{request.node.name}"[:60]
    dense = ChromaDense(
        host="localhost", port=8001, collection=collection, embed_query=toy_embed
    )
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


def test_reupserting_the_same_chunk_does_not_duplicate_it(store):
    store.upsert(CORPUS, [toy_embed(c.text) for c in CORPUS])
    store.upsert(CORPUS, [toy_embed(c.text) for c in CORPUS])

    results = store.search("hypertension", top_k=10)

    assert len(results) == len(set(results)) == 3
