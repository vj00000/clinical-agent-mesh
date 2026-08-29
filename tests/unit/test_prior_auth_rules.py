"""The deterministic half of prior auth: the verdict, and the words for it.

The model marks criteria. It does not decide. These tests pin the rule exactly,
because "the prompt seemed to handle it" is not a coverage policy.
"""

from mesh.agents.prior_auth_rules import (
    Criterion,
    Decision,
    decide_coverage,
    render_decision,
)


def _criterion(text: str, met: bool | None) -> Criterion:
    return Criterion(text=text, met=met, chunk_id="c1")


def test_every_criterion_met_is_an_approval():
    criteria = [
        _criterion("documented trial of metformin", True),
        _criterion("HbA1c above 7.5", True),
    ]

    assert decide_coverage(criteria) is Decision.APPROVE


def test_one_unmet_criterion_denies():
    criteria = [
        _criterion("documented trial of metformin", True),
        _criterion("HbA1c above 7.5", False),
    ]

    assert decide_coverage(criteria) is Decision.DENY


def test_an_unresolved_criterion_asks_rather_than_guesses():
    """None means the request is silent, not that the requirement failed."""
    criteria = [
        _criterion("documented trial of metformin", True),
        _criterion("HbA1c above 7.5", None),
    ]

    assert decide_coverage(criteria) is Decision.MORE_INFO


def test_an_unmet_criterion_outranks_an_unresolved_one():
    """More information cannot rescue a requirement that has already failed."""
    criteria = [
        _criterion("HbA1c above 7.5", False),
        _criterion("documented trial of metformin", None),
    ]

    assert decide_coverage(criteria) is Decision.DENY


def test_no_criteria_is_not_an_approval():
    """No policy was retrieved to judge against. Approving on an empty checklist
    is the single worst bug this module could have."""
    assert decide_coverage([]) is Decision.MORE_INFO


def test_the_answer_lists_every_criterion_and_how_it_was_marked():
    criteria = [
        _criterion("documented trial of metformin", True),
        _criterion("HbA1c above 7.5", None),
    ]

    rendered = render_decision(Decision.MORE_INFO, criteria)

    assert "[met] documented trial of metformin" in rendered
    assert "[unresolved] HbA1c above 7.5" in rendered


def test_a_decision_with_no_criteria_still_reads_as_a_sentence():
    rendered = render_decision(Decision.MORE_INFO, [])

    assert rendered.strip() != ""
    assert "\n" not in rendered
