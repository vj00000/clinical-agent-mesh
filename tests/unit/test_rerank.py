"""Cross-encoder reranking.

The scorer is injected so the ordering logic is tested without importing torch.

Note the contrast with retrieval: an unreachable vector store raises, but a failed
reranker falls back to fusion order. Reranking only *reorders* evidence that was
already retrieved and grounded, so losing it degrades answer quality. Losing
retrieval changes what evidence exists at all, which is a correctness problem.
Degrading is right in one case and wrong in the other.
"""

from mesh.retrieval.chunking import Chunk
from mesh.retrieval.rerank import Reranker


def _chunk(chunk_id: str, text: str = "text") -> Chunk:
    return Chunk(chunk_id=chunk_id, source="test", text=text, ordinal=0)


CANDIDATES = [_chunk("c1"), _chunk("c2"), _chunk("c3"), _chunk("c4")]


class RecordingScorer:
    """Scores by a fixed lookup and records how many batches it received."""

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.batches = 0
        self.pairs_seen = 0

    def __call__(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.batches += 1
        self.pairs_seen += len(pairs)
        return [self.scores.get(text, 0.0) for _, text in pairs]


def test_candidates_are_reordered_by_score():
    scorer = RecordingScorer({"c1": 0.1, "c2": 0.9, "c3": 0.5, "c4": 0.2})
    reranker = Reranker(score_pairs=scorer)

    result = reranker.rerank("q", [_chunk(c, c) for c in ("c1", "c2", "c3", "c4")], top_n=4)

    assert [c.chunk_id for c in result] == ["c2", "c3", "c4", "c1"]


def test_only_the_top_n_survive():
    """The point of reranking: hand the model 5 strong chunks, not 20 mixed ones."""
    scorer = RecordingScorer({"c1": 0.1, "c2": 0.9, "c3": 0.5, "c4": 0.2})
    reranker = Reranker(score_pairs=scorer)

    result = reranker.rerank("q", [_chunk(c, c) for c in ("c1", "c2", "c3", "c4")], top_n=2)

    assert [c.chunk_id for c in result] == ["c2", "c3"]


def test_all_pairs_are_scored_in_a_single_batch():
    """Per-candidate calls would make reranking slower than the LLM it feeds."""
    scorer = RecordingScorer({})
    reranker = Reranker(score_pairs=scorer)

    reranker.rerank("q", CANDIDATES, top_n=2)

    assert scorer.batches == 1
    assert scorer.pairs_seen == 4


def test_equal_scores_keep_their_incoming_order():
    """Fusion order is meaningful, so ties must not be shuffled."""
    scorer = RecordingScorer({})
    reranker = Reranker(score_pairs=scorer)

    result = reranker.rerank("q", CANDIDATES, top_n=4)

    assert [c.chunk_id for c in result] == ["c1", "c2", "c3", "c4"]


def test_no_candidates_needs_no_scoring():
    scorer = RecordingScorer({})
    reranker = Reranker(score_pairs=scorer)

    assert reranker.rerank("q", [], top_n=5) == []
    assert scorer.batches == 0


def test_a_failing_scorer_falls_back_to_the_incoming_order():
    """Unlike retrieval, this degrades: the evidence is still grounded, just less
    well ordered, so a reranker outage must not fail the whole query."""

    def broken(pairs: list[tuple[str, str]]) -> list[float]:
        raise RuntimeError("model failed to load")

    reranker = Reranker(score_pairs=broken)

    result = reranker.rerank("q", CANDIDATES, top_n=2)

    assert [c.chunk_id for c in result] == ["c1", "c2"]


def test_the_query_is_paired_with_every_candidate():
    seen: list[tuple[str, str]] = []

    def scorer(pairs: list[tuple[str, str]]) -> list[float]:
        seen.extend(pairs)
        return [0.0] * len(pairs)

    Reranker(score_pairs=scorer).rerank("what lowers blood pressure?", CANDIDATES, top_n=4)

    assert {q for q, _ in seen} == {"what lowers blood pressure?"}
    assert len(seen) == 4
