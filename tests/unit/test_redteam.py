"""The adversarial cases that need no model.

Injection and PHI are settled by guard_in, which is deterministic, so these are
real red-team results produced by the ordinary test run rather than claims
waiting on a paid evaluation.
"""

import pytest

from mesh.evals.redteam import (
    DETERMINISTIC_CATEGORIES,
    blocks_injection,
    load_redteam_cases,
    redacts_identifiers,
)

CASES = load_redteam_cases()

INJECTION = [c for c in CASES if c.category == "injection"]
PHI = [c for c in CASES if c.category == "phi"]


def test_the_suite_loads_and_is_not_empty():
    assert len(CASES) >= 15


def test_every_case_carries_a_category_we_know_how_to_score():
    """A case nobody scores is a case that silently passes forever."""
    known = DETERMINISTIC_CATEGORIES | {"unsafe_dosing", "out_of_scope", "ungrounded_pressure"}

    assert {case.category for case in CASES} <= known


@pytest.mark.parametrize("case", INJECTION, ids=lambda c: c.query[:40])
def test_every_injection_attempt_is_blocked_before_the_supervisor(case):
    """guard_in routes these straight to refuse, so the classifier never sees
    the payload. That is the property, not merely that the answer looks safe."""
    assert blocks_injection(case.query)


@pytest.mark.parametrize("case", PHI, ids=lambda c: c.query[:40])
def test_every_identifier_is_stripped_before_anything_downstream(case):
    assert redacts_identifiers(case.query)


@pytest.mark.parametrize("case", PHI, ids=lambda c: c.query[:40])
def test_a_redacted_query_is_still_answerable(case):
    """Redaction must not destroy the question. Stripping the clinical content
    along with the identifiers would turn every PHI-bearing query into a
    refusal, which looks like safety and is actually a bug."""
    from mesh.guardrails.phi import redact_phi

    assert redact_phi(case.query).text.strip()
