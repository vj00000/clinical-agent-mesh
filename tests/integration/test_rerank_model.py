"""The real cross-encoder, end to end.

The unit tests prove the ordering logic with an injected scorer. This proves the
factory actually loads a model and that the model actually discriminates — the
part a stub can never tell you.

Marked `rerank` and skipped unless the optional extra is installed, so a
torch-free install and CI stay green. First run downloads the model (~90MB).
"""

import pytest

from mesh.retrieval.chunking import Chunk
from mesh.retrieval.rerank import Reranker, build_cross_encoder

sentence_transformers = pytest.importorskip(
    "sentence_transformers", reason="needs the `rerank` extra: uv sync --extra rerank"
)

pytestmark = pytest.mark.rerank


@pytest.fixture(scope="module")
def reranker() -> Reranker:
    return Reranker(score_pairs=build_cross_encoder())


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, source="test", text=text, ordinal=0)


def test_the_model_ranks_a_relevant_passage_above_an_irrelevant_one(reranker):
    candidates = [
        _chunk("irrelevant", "The Amazon rainforest spans nine countries in South America."),
        _chunk(
            "relevant",
            "Thiazide diuretics are recommended as first-line therapy for hypertension.",
        ),
    ]

    result = reranker.rerank(
        "What is first-line treatment for high blood pressure?", candidates, top_n=2
    )

    assert [c.chunk_id for c in result] == ["relevant", "irrelevant"]


def test_the_model_rescues_a_relevant_chunk_that_fusion_ranked_last(reranker):
    """This is the whole justification for the rerank stage: fusion puts a strong
    chunk at position 4, and the cross-encoder pulls it to the front."""
    candidates = [
        _chunk("c1", "Blood pressure should be measured after five minutes of seated rest."),
        _chunk("c2", "Cuff size affects the accuracy of a blood pressure reading."),
        _chunk("c3", "Home blood pressure monitors should be validated annually."),
        _chunk("c4", "For stage 2 hypertension, start two first-line agents concurrently."),
    ]

    result = reranker.rerank(
        "How should I start treatment for stage 2 hypertension?", candidates, top_n=1
    )

    assert [c.chunk_id for c in result] == ["c4"]


def test_reranking_a_realistic_candidate_set_stays_under_a_second(reranker):
    """Sanity check on the cost claim: a local cross-encoder over ~20 candidates
    must not dominate the latency of the LLM call it feeds."""
    import time

    candidates = [
        _chunk(f"c{i}", f"Clinical guidance passage number {i} about hypertension.")
        for i in range(20)
    ]

    start = time.monotonic()
    reranker.rerank("hypertension first-line therapy", candidates, top_n=5)
    elapsed = time.monotonic() - start

    assert elapsed < 1.0, f"reranking 20 candidates took {elapsed:.2f}s"
