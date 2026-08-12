"""Confidence gating for the supervisor.

Separated from the LLM call so the policy — when to answer, when to ask, when to
refuse — is testable without a network round trip.
"""

from mesh.agents.routing import Classification, decide_route


def test_a_confident_classification_is_routed_to_its_specialist():
    decision = decide_route(
        Classification(route="guideline", confidence=0.9, rationale="asks about therapy"),
        threshold=0.6,
    )

    assert decision == "guideline"


def test_a_low_confidence_classification_asks_for_clarification_instead_of_guessing():
    decision = decide_route(
        Classification(route="guideline", confidence=0.3, rationale="ambiguous"),
        threshold=0.6,
    )

    assert decision == "clarify"


def test_confidence_exactly_at_the_threshold_is_accepted():
    decision = decide_route(
        Classification(route="triage", confidence=0.6, rationale="symptom report"),
        threshold=0.6,
    )

    assert decision == "triage"


def test_an_out_of_scope_classification_is_refused_however_confident_it_is():
    decision = decide_route(
        Classification(route="refuse", confidence=0.99, rationale="asks for stock tips"),
        threshold=0.6,
    )

    assert decision == "refuse"
