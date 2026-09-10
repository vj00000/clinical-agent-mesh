"""Turning a model's reported citations into grounded ones.

Three specialists ask a model to cite chunk ids and quote the sentence it used.
In every case the model supplies the id and the quote, and the source is looked
up from the chunk it named: a model cannot invent provenance it was never given.
"""

from pydantic import BaseModel, Field

from mesh.retrieval.chunking import Chunk
from mesh.state import Citation

# Provenance for a citation naming a chunk that was never retrieved. Such a
# citation still has to survive as a Citation so the verifier can reject it --
# dropping it here would hide the failure instead of catching it -- and `source`
# is a required field, so it needs a non-empty placeholder.
UNKNOWN_SOURCE = "unretrieved"


class ClaimCitation(BaseModel):
    """What a model returns: the chunk it used, and the sentence it relied on."""

    chunk_id: str = Field(min_length=1)
    quote: str


def format_chunks(chunks: list[Chunk]) -> str:
    """Render passages with their ids, so the model has something to cite."""
    return "\n\n".join(f"[{chunk.chunk_id}] {chunk.text}" for chunk in chunks)


def to_citations(cited: list[ClaimCitation], chunks: list[Chunk]) -> list[Citation]:
    """Attach each cited chunk's real source, or the placeholder if invented."""
    source_by_id = {chunk.chunk_id: chunk.source for chunk in chunks}

    return [
        Citation(
            chunk_id=claim.chunk_id,
            source=source_by_id.get(claim.chunk_id, UNKNOWN_SOURCE),
            quote=claim.quote,
        )
        for claim in cited
    ]
