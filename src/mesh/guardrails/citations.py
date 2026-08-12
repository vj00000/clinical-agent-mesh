"""Citation verification for guard_out."""

from collections.abc import Collection, Sequence

from pydantic import BaseModel

from mesh.state import Citation


class CitationVerdict(BaseModel):
    grounded: bool
    unsupported: list[str]


def verify_citations(
    citations: Sequence[Citation], *, retrieved: Collection[str]
) -> CitationVerdict:
    """Check that every citation points at a chunk retrieved this turn.

    An answer with no citations is not grounded either: there is nothing to check
    it against, which for a clinical claim is indistinguishable from a guess.
    """
    unsupported = sorted({c.chunk_id for c in citations if c.chunk_id not in retrieved})

    return CitationVerdict(
        grounded=bool(citations) and not unsupported,
        unsupported=unsupported,
    )
