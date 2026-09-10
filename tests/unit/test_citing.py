"""The model-reported-citation mapping shared by three specialists."""

from mesh.agents.citing import (
    UNKNOWN_SOURCE,
    ClaimCitation,
    format_chunks,
    to_citations,
)
from mesh.retrieval.chunking import Chunk

CHUNKS = [
    Chunk(chunk_id="c1", source="cdc-htn", text="thiazides are first-line", ordinal=0),
    Chunk(chunk_id="c2", source="pmid:123", text="renal dosing differs", ordinal=1),
]


def test_a_citation_gets_the_source_of_the_chunk_it_names():
    """The model supplies the id and the quote; the code supplies provenance."""
    citations = to_citations([ClaimCitation(chunk_id="c2", quote="...")], CHUNKS)

    assert citations[0].source == "pmid:123"


def test_a_citation_naming_an_unretrieved_chunk_survives_with_a_placeholder():
    """It has to reach the verifier to be rejected. Dropping it here would hide
    the invention rather than catch it."""
    citations = to_citations([ClaimCitation(chunk_id="ghost", quote="...")], CHUNKS)

    assert [c.chunk_id for c in citations] == ["ghost"]
    assert citations[0].source == UNKNOWN_SOURCE


def test_citing_nothing_yields_nothing():
    assert to_citations([], CHUNKS) == []


def test_passages_are_rendered_with_their_ids():
    """Without the id in front of the text the model has nothing to cite by."""
    rendered = format_chunks(CHUNKS)

    assert "[c1] thiazides are first-line" in rendered
    assert "[c2] renal dosing differs" in rendered
