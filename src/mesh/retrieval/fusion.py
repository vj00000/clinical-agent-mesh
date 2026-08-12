"""Reciprocal rank fusion of several ranked result lists.

Dense similarity and BM25 produce scores on incomparable scales, so they are
merged by rank rather than by score.
"""

from collections.abc import Sequence

# Standard damping constant from the original RRF paper; keeps a single list's
# top hit from dominating a document that ranks moderately well in both lists.
RRF_K = 60


def reciprocal_rank_fusion(ranked_lists: Sequence[Sequence[str]], *, k: int = RRF_K) -> list[str]:
    """Merge ranked id lists into one, best first."""
    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}

    position = 0
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
            if doc_id not in first_seen:
                first_seen[doc_id] = position
                position += 1

    return sorted(scores, key=lambda doc_id: (-scores[doc_id], first_seen[doc_id]))
