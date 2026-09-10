"""The composition root's assembly decisions.

Building real retrieval needs Chroma, so that is covered by the integration
tests. What is checked here is the reranker fallback, which has to behave
correctly on a machine without the optional torch extra installed.
"""

from mesh.composition import _no_reranking, build_reranker
from mesh.retrieval.chunking import Chunk
from mesh.retrieval.rerank import Reranker


def _chunk(chunk_id: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, source="test", text=f"passage {chunk_id}", ordinal=0)


FUSED = [_chunk("c1"), _chunk("c2"), _chunk("c3")]


def test_the_no_op_scorer_leaves_the_fusion_order_untouched():
    """Fusion already ranked these. A scorer that reordered them would be worse
    than no reranking at all."""
    reranker = Reranker(score_pairs=_no_reranking)

    ordered = reranker.rerank("q", FUSED, top_n=3)

    assert [chunk.chunk_id for chunk in ordered] == ["c1", "c2", "c3"]


def test_the_no_op_scorer_still_honours_top_n():
    reranker = Reranker(score_pairs=_no_reranking)

    assert len(reranker.rerank("q", FUSED, top_n=2)) == 2


def test_a_reranker_is_always_returned():
    """Missing the optional extra degrades answer quality; it must not stop the
    system from starting, because reranking only reorders evidence that is
    already retrieved and grounded."""
    assert isinstance(build_reranker(), Reranker)
