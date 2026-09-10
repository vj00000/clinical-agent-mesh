"""Metrics for the evaluation harness.

Pure functions over recorded runs, so the numbers can be recomputed from stored
outcomes without re-spending on the API, and so the metrics themselves are
unit-tested rather than trusted.

Faithfulness is the exception and is deliberately shaped differently: judging
whether an answer is actually supported by its sources needs a model, so the
judge is injected. Everything else here is arithmetic.
"""

from collections.abc import Callable, Sequence

from pydantic import BaseModel, Field

from mesh.state import Citation


class GoldenCase(BaseModel):
    """One labelled question.

    `answerable` is the important label. Roughly a quarter of the set is
    deliberately unanswerable from the corpus, because a system that answers
    everything confidently scores well on every other metric while being
    useless -- calibrated refusal is the thing worth measuring.
    """

    question: str = Field(min_length=1)
    answerable: bool
    route: str = Field(min_length=1)


class RunOutcome(BaseModel):
    """What the mesh did with one case."""

    case: GoldenCase
    route: str
    answer: str
    citations: list[Citation]
    retrieved_ids: list[str]
    refused: bool


class RefusalReport(BaseModel):
    total: int
    correct: int
    accuracy: float
    # Answered something it had no evidence for. The dangerous direction.
    answered_when_unanswerable: int
    # Refused something it could have answered. Annoying, not dangerous.
    refused_when_answerable: int


def refusal_correctness(outcomes: Sequence[RunOutcome]) -> RefusalReport:
    """Did it refuse exactly when it should have?

    The two error directions are counted separately on purpose. They are not
    equally bad: an unnecessary refusal wastes someone's time, while an answer
    invented for a question the corpus cannot support is the failure this whole
    system exists to prevent.
    """
    if not outcomes:
        raise ValueError("cannot score an empty run: the result would be meaningless")

    answered_when_unanswerable = sum(
        1 for outcome in outcomes if not outcome.case.answerable and not outcome.refused
    )
    refused_when_answerable = sum(
        1 for outcome in outcomes if outcome.case.answerable and outcome.refused
    )
    correct = len(outcomes) - answered_when_unanswerable - refused_when_answerable

    return RefusalReport(
        total=len(outcomes),
        correct=correct,
        accuracy=correct / len(outcomes),
        answered_when_unanswerable=answered_when_unanswerable,
        refused_when_answerable=refused_when_answerable,
    )


def citation_accuracy(outcomes: Sequence[RunOutcome]) -> float:
    """Fraction of citations naming a chunk that was actually retrieved.

    Cases that produced no citations are skipped rather than scored zero: a
    refusal cites nothing by design, and counting that as a citation failure
    would punish the system for behaving correctly.
    """
    cited = [outcome for outcome in outcomes if outcome.citations]
    if not cited:
        return 0.0

    total = 0
    grounded = 0
    for outcome in cited:
        retrieved = set(outcome.retrieved_ids)
        for citation in outcome.citations:
            total += 1
            grounded += citation.chunk_id in retrieved

    return grounded / total


def answer_rate(outcomes: Sequence[RunOutcome]) -> float:
    """Fraction of cases that produced an answer rather than a refusal."""
    if not outcomes:
        return 0.0

    return sum(1 for outcome in outcomes if not outcome.refused) / len(outcomes)


def routing_accuracy(outcomes: Sequence[RunOutcome]) -> float:
    """Fraction of cases that reached the specialist their label names."""
    if not outcomes:
        return 0.0

    return sum(1 for outcome in outcomes if outcome.route == outcome.case.route) / len(outcomes)


# Given an answer and its citations, is every claim actually supported?
Judge = Callable[[str, list[Citation]], bool]


def faithfulness(outcomes: Sequence[RunOutcome], *, judge: Judge) -> float:
    """Fraction of answered cases whose claims the judge finds supported.

    Injected rather than built in, for the same reason every other model call in
    this codebase is: the metric is then testable without a key, and the judge
    can be swapped without touching the metric.

    Only answered cases are judged. A refusal asserts nothing, so there is
    nothing for a judge to find unsupported, and scoring it would make refusing
    everything the way to a perfect score.
    """
    answered = [outcome for outcome in outcomes if not outcome.refused]
    if not answered:
        return 0.0

    return sum(1 for o in answered if judge(o.answer, o.citations)) / len(answered)
