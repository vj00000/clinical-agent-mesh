"""Routing benchmark scoring.

Accuracy alone hides which agents get confused for each other, and that pairing
is what actually tells you which prompt to fix. So the report carries a confusion
matrix and the individual misroutes.
"""

import pytest

from mesh.agents.routing import SPECIALISTS, Classification
from mesh.evals.routing import RoutingCase, load_routing_cases, score_routing


def _cases() -> list[RoutingCase]:
    return [
        RoutingCase(query="first-line therapy for hypertension?", expected="guideline"),
        RoutingCase(query="crushing chest pain right now", expected="triage"),
        RoutingCase(query="is this infusion covered by Medicare?", expected="prior_auth"),
        RoutingCase(query="what are my discharge medications?", expected="discharge"),
    ]


def _always(route: str):
    def classify(query: str) -> Classification:
        return Classification(route=route, confidence=0.9, rationale="stub")

    return classify


def _perfect(cases: list[RoutingCase]):
    lookup = {case.query: case.expected for case in cases}

    def classify(query: str) -> Classification:
        return Classification(route=lookup[query], confidence=0.9, rationale="stub")

    return classify


def test_a_perfect_classifier_scores_one():
    cases = _cases()

    report = score_routing(cases, _perfect(cases))

    assert report.accuracy == 1.0
    assert report.misroutes == []


def test_accuracy_is_the_fraction_of_correct_routes():
    report = score_routing(_cases(), _always("guideline"))

    assert report.accuracy == 0.25


def test_the_confusion_matrix_records_expected_against_predicted():
    report = score_routing(_cases(), _always("guideline"))

    assert report.confusion[("triage", "guideline")] == 1
    assert report.confusion[("guideline", "guideline")] == 1


def test_each_misroute_is_listed_for_inspection():
    report = score_routing(_cases(), _always("guideline"))

    misrouted_queries = {m.query for m in report.misroutes}
    assert "crushing chest pain right now" in misrouted_queries
    assert len(report.misroutes) == 3


def test_a_classifier_that_errors_counts_as_a_miss_rather_than_crashing_the_run():
    """One bad response must not lose the whole benchmark."""

    def broken(query: str) -> Classification:
        raise RuntimeError("rate limited")

    report = score_routing(_cases(), broken)

    assert report.accuracy == 0.0
    assert len(report.misroutes) == 4


def test_scoring_no_cases_is_an_error_rather_than_a_perfect_score():
    with pytest.raises(ValueError):
        score_routing([], _always("guideline"))


# --- integrity of the shipped benchmark itself ---------------------------------
#
# A benchmark with a typo in an expected label reports a permanent failure that
# looks like a model problem, so the dataset is checked like any other input.


def test_the_shipped_benchmark_loads():
    cases = load_routing_cases()

    assert len(cases) >= 30


def test_every_expected_label_is_a_real_route():
    valid = SPECIALISTS | {"refuse"}

    assert {case.expected for case in load_routing_cases()} <= valid


def test_no_query_appears_twice():
    """A duplicate silently double-weights whatever it tests."""
    queries = [case.query for case in load_routing_cases()]

    assert len(queries) == len(set(queries))


def test_every_route_has_cases():
    covered = {case.expected for case in load_routing_cases()}

    assert covered == SPECIALISTS | {"refuse"}
