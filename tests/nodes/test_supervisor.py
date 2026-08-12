"""The supervisor node.

The classifier is injected, so the routing policy — accept, clarify, refuse, and
what to do when the model itself fails — is verified without an LLM call.
"""

from mesh.agents.routing import Classification
from mesh.agents.supervisor import make_supervisor
from mesh.state import new_state


def _classifier(route: str, confidence: float):
    def classify(query: str) -> Classification:
        return Classification(route=route, confidence=confidence, rationale="stub")

    return classify


def test_a_confident_classification_is_routed_to_its_specialist():
    supervisor = make_supervisor(_classifier("guideline", 0.9), threshold=0.6)

    update = supervisor(state=new_state("what is first-line therapy for hypertension?"))

    assert update["route"] == "guideline"
    assert update["confidence"] == 0.9


def test_a_low_confidence_classification_is_sent_to_clarify():
    supervisor = make_supervisor(_classifier("guideline", 0.3), threshold=0.6)

    update = supervisor(state=new_state("what about the other one"))

    assert update["route"] == "clarify"


def test_the_original_confidence_is_preserved_when_clarifying():
    """The gate changes the route, not the record of how unsure the model was."""
    supervisor = make_supervisor(_classifier("guideline", 0.3), threshold=0.6)

    update = supervisor(state=new_state("q"))

    assert update["confidence"] == 0.3


def test_an_out_of_scope_classification_is_refused_however_confident():
    supervisor = make_supervisor(_classifier("refuse", 0.99), threshold=0.6)

    update = supervisor(state=new_state("what stocks should I buy?"))

    assert update["route"] == "refuse"


def test_the_classifier_receives_the_redacted_query():
    """guard_in has already run; the model must never see the raw identifiers."""
    seen: list[str] = []

    def classify(query: str) -> Classification:
        seen.append(query)
        return Classification(route="guideline", confidence=0.9, rationale="stub")

    supervisor = make_supervisor(classify, threshold=0.6)
    state = new_state("redacted [EMAIL] question about blood pressure")
    supervisor(state=state)

    assert seen == ["redacted [EMAIL] question about blood pressure"]


def test_a_failing_classifier_refuses_rather_than_guessing_a_route():
    """Fail closed, as retrieval does: a broken classifier must not pick a
    specialist at random and present its answer as grounded."""

    def broken(query: str) -> Classification:
        raise RuntimeError("rate limited")

    supervisor = make_supervisor(broken, threshold=0.6)

    update = supervisor(state=new_state("what is first-line therapy?"))

    assert update["route"] == "refuse"
    assert update["confidence"] == 0.0
    assert "classifier_error" in update["guard_flags"]
