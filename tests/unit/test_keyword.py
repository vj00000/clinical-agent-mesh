"""BM25 keyword retrieval.

This exists because dense embeddings blur exact tokens: drug names, ICD codes,
and dosages are precisely where vector-only retrieval loses.
"""

from mesh.retrieval.chunking import Chunk
from mesh.retrieval.keyword import BM25Index


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, source="test", text=text, ordinal=0)


CORPUS = [
    _chunk("c1", "Lisinopril is an ACE inhibitor used for hypertension management."),
    _chunk("c2", "Metformin is first-line pharmacotherapy for type 2 diabetes."),
    _chunk("c3", "Blood pressure should be measured after five minutes of rest."),
]


def test_an_exact_drug_name_retrieves_its_chunk_first():
    results = BM25Index(CORPUS).search("lisinopril", top_k=3)

    assert results[0] == "c1"


def test_matching_ignores_case():
    results = BM25Index(CORPUS).search("METFORMIN", top_k=3)

    assert results[0] == "c2"


def test_top_k_limits_the_number_of_results():
    results = BM25Index(CORPUS).search("hypertension pressure diabetes", top_k=2)

    assert len(results) == 2


def test_a_query_sharing_no_terms_with_the_corpus_returns_nothing():
    results = BM25Index(CORPUS).search("xyzzy quux", top_k=3)

    assert results == []


def test_searching_an_empty_corpus_returns_nothing():
    results = BM25Index([]).search("lisinopril", top_k=3)

    assert results == []
