"""Citation verification for guard_out.

This is the deterministic half of grounding: a citation must point at a chunk the
retriever actually returned this turn. A model can invent a plausible chunk id as
easily as it can invent a fact, and an invented citation is worse than none —
it looks like evidence.

Judging whether the cited chunk genuinely *supports* the claim needs a model and
lives in the guideline subgraph. This check runs first because it is free and
catches the crudest failure.
"""

from mesh.guardrails.citations import verify_citations
from mesh.state import Citation


def _citation(chunk_id: str) -> Citation:
    return Citation(chunk_id=chunk_id, source="cdc-htn", quote="First-line therapy is...")


def test_citations_pointing_at_retrieved_chunks_are_accepted():
    verdict = verify_citations([_citation("c1"), _citation("c2")], retrieved={"c1", "c2", "c3"})

    assert verdict.grounded
    assert verdict.unsupported == []


def test_a_citation_pointing_at_an_unretrieved_chunk_is_rejected():
    """The id was never returned this turn, so the model invented it."""
    verdict = verify_citations([_citation("c1"), _citation("ghost")], retrieved={"c1"})

    assert not verdict.grounded
    assert verdict.unsupported == ["ghost"]


def test_an_answer_with_no_citations_at_all_is_rejected():
    verdict = verify_citations([], retrieved={"c1", "c2"})

    assert not verdict.grounded


def test_repeating_a_citation_is_allowed():
    """Several claims may legitimately rest on the same chunk."""
    verdict = verify_citations([_citation("c1"), _citation("c1")], retrieved={"c1"})

    assert verdict.grounded


def test_every_invented_citation_is_reported_once():
    verdict = verify_citations(
        [_citation("ghost"), _citation("ghost"), _citation("phantom")], retrieved={"c1"}
    )

    assert sorted(verdict.unsupported) == ["ghost", "phantom"]


def test_nothing_retrieved_means_nothing_can_be_grounded():
    verdict = verify_citations([_citation("c1")], retrieved=set())

    assert not verdict.grounded
    assert verdict.unsupported == ["c1"]
