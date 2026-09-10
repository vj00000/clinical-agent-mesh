"""The evaluation metrics.

Arithmetic over recorded outcomes, so all of this is testable with no model and
no network. The judge is injected for the one metric that needs an opinion.
"""

import pytest

from mesh.evals.metrics import (
    GoldenCase,
    RunOutcome,
    answer_rate,
    citation_accuracy,
    faithfulness,
    refusal_correctness,
    routing_accuracy,
)
from mesh.state import Citation


def _outcome(
    *,
    answerable: bool = True,
    refused: bool = False,
    route: str = "guideline",
    expected_route: str = "guideline",
    citations: list[Citation] | None = None,
    retrieved_ids: list[str] | None = None,
) -> RunOutcome:
    return RunOutcome(
        case=GoldenCase(question="q", answerable=answerable, route=expected_route),
        route=route,
        answer="an answer",
        citations=citations or [],
        retrieved_ids=retrieved_ids or ["c1"],
        refused=refused,
    )


CITED = Citation(chunk_id="c1", source="cdc", quote="...")
INVENTED = Citation(chunk_id="ghost", source="cdc", quote="...")


def test_refusing_the_unanswerable_and_answering_the_rest_scores_one():
    outcomes = [
        _outcome(answerable=True, refused=False),
        _outcome(answerable=False, refused=True),
    ]

    assert refusal_correctness(outcomes).accuracy == 1.0


def test_the_two_error_directions_are_counted_separately():
    """They are not equally bad. Answering without evidence is the failure this
    system exists to prevent; refusing unnecessarily just wastes time."""
    outcomes = [
        _outcome(answerable=False, refused=False),
        _outcome(answerable=True, refused=True),
    ]

    report = refusal_correctness(outcomes)

    assert report.answered_when_unanswerable == 1
    assert report.refused_when_answerable == 1
    assert report.accuracy == 0.0


def test_scoring_an_empty_run_is_refused():
    """An empty run has no meaningful accuracy, and returning 1.0 or 0.0 would
    let a broken harness pass or fail the CI gate for the wrong reason."""
    with pytest.raises(ValueError):
        refusal_correctness([])


def test_citation_accuracy_counts_citations_not_cases():
    outcomes = [_outcome(citations=[CITED, INVENTED], retrieved_ids=["c1"])]

    assert citation_accuracy(outcomes) == 0.5


def test_cases_that_cite_nothing_are_skipped_not_scored_zero():
    """A refusal cites nothing by design. Scoring that as a citation failure
    would punish the system for behaving correctly."""
    outcomes = [
        _outcome(citations=[CITED], retrieved_ids=["c1"]),
        _outcome(refused=True, citations=[]),
    ]

    assert citation_accuracy(outcomes) == 1.0


def test_answer_rate_is_the_share_that_did_not_refuse():
    outcomes = [_outcome(refused=False), _outcome(refused=True)]

    assert answer_rate(outcomes) == 0.5


def test_routing_accuracy_compares_the_route_taken_with_the_label():
    outcomes = [
        _outcome(route="guideline", expected_route="guideline"),
        _outcome(route="triage", expected_route="discharge"),
    ]

    assert routing_accuracy(outcomes) == 0.5


def test_faithfulness_asks_the_judge_about_answered_cases():
    outcomes = [_outcome(citations=[CITED]), _outcome(citations=[CITED])]

    assert faithfulness(outcomes, judge=lambda answer, citations: True) == 1.0
    assert faithfulness(outcomes, judge=lambda answer, citations: False) == 0.0


def test_refusals_are_not_judged_for_faithfulness():
    """A refusal asserts nothing, so there is nothing to find unsupported. If
    refusals counted, refusing everything would be the way to a perfect score."""
    outcomes = [_outcome(refused=True), _outcome(refused=False)]

    assert faithfulness(outcomes, judge=lambda answer, citations: True) == 1.0
