"""The evaluation harness, driven by a stub mesh.

The harness is the thing that decides whether a build passes, so it is tested
rather than trusted -- and because the mesh is injected, that costs nothing.
"""

from typing import Any

from mesh.evals.harness import (
    THRESHOLDS,
    EvalReport,
    is_refusal,
    load_golden_cases,
    run_cases,
    score,
)
from mesh.evals.metrics import GoldenCase
from mesh.graph import OUT_OF_SCOPE_REFUSAL
from mesh.state import Citation


class StubMesh:
    """Returns a canned final state, or raises for one nominated question."""

    def __init__(self, state: dict[str, Any], *, raises_on: str | None = None) -> None:
        self.state = state
        self.raises_on = raises_on

    def invoke(self, initial: dict[str, Any]) -> dict[str, Any]:
        if self.raises_on and initial["query"] == self.raises_on:
            raise RuntimeError("chroma is down")

        return {**self.state, "query": initial["query"]}


ANSWERED = {
    "route": "guideline",
    "answer": "Thiazides are first-line.",
    "citations": [Citation(chunk_id="c1", source="cdc", quote="...")],
    "retrieved_ids": ["c1", "c2"],
}

CASE = GoldenCase(question="what is first-line therapy?", answerable=True, route="guideline")


def test_the_shipped_golden_set_loads_and_is_labelled():
    cases = load_golden_cases()

    assert len(cases) >= 60
    # Calibrated refusal is unmeasurable without cases that should be refused.
    assert any(not case.answerable for case in cases)
    assert {case.route for case in cases} == {"guideline", "triage", "prior_auth", "discharge"}


def test_a_known_refusal_string_counts_as_a_refusal():
    assert is_refusal(OUT_OF_SCOPE_REFUSAL, "refuse")


def test_a_clarifying_question_counts_as_not_answering():
    """Asking beats guessing, so this is not a failure -- but it is not an
    answer, and counting it as one would let a system that always asks for
    clarification look perfectly calibrated."""
    assert is_refusal("Could you say more?", "clarify")


def test_a_real_answer_is_not_a_refusal():
    assert not is_refusal("Thiazides are first-line.", "guideline")


def test_a_case_that_raises_is_recorded_as_a_miss_not_a_crash():
    """One bad case must not cost the whole run. On a paid API that is money."""
    mesh = StubMesh(ANSWERED, raises_on=CASE.question)

    outcomes = run_cases(mesh, [CASE])

    assert outcomes[0].route == "__error__"
    assert outcomes[0].refused is True


def test_outcomes_carry_what_the_metrics_need():
    outcomes = run_cases(StubMesh(ANSWERED), [CASE])

    assert outcomes[0].citations[0].chunk_id == "c1"
    assert outcomes[0].retrieved_ids == ["c1", "c2"]
    assert outcomes[0].refused is False


def test_a_perfect_run_reports_perfect_scores():
    report = score(run_cases(StubMesh(ANSWERED), [CASE]))

    assert report.refusal_correctness == 1.0
    assert report.citation_accuracy == 1.0
    assert report.routing_accuracy == 1.0
    assert report.failures() == []


def test_a_run_below_threshold_names_the_metric_that_failed():
    """The gate has to say which metric broke, or the next person disables it
    rather than debug it."""
    report = EvalReport(
        cases=10,
        refusal_correctness=0.10,
        citation_accuracy=1.0,
        routing_accuracy=1.0,
        answer_rate=1.0,
        answered_when_unanswerable=9,
        refused_when_answerable=0,
    )

    failures = report.failures()

    assert len(failures) == 1
    assert "refusal_correctness" in failures[0]


def test_every_threshold_is_reachable():
    """A threshold above 1.0 could never pass, and a gate that can never pass
    gets deleted rather than met."""
    assert all(0.0 < floor <= 1.0 for floor in THRESHOLDS.values())
