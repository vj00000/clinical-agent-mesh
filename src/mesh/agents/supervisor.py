"""The supervisor node: classify the query, then decide whether to trust it.

The classifier is a plain callable so this module has no dependency on any model
provider, and the routing policy is testable without a network call. The LLM-backed
classifier is built separately in `build_classifier`.
"""

from collections.abc import Callable
from typing import Any

from langchain_core.language_models import BaseChatModel

from mesh.agents.routing import Classification, decide_route
from mesh.state import MeshState

Classifier = Callable[[str], Classification]

SYSTEM_PROMPT = """You route clinical questions to one specialist.

Routes:
- guideline: evidence questions about therapy, diagnosis, or management
- triage: someone describing symptoms and asking what to do now
- prior_auth: coverage, payer policy, or authorisation criteria
- discharge: medications, discharge summaries, or drug interactions
- refuse: anything outside clinical care

Report your genuine confidence. A low score is more useful than a confident
guess, because a low score makes the system ask the user what they meant."""


def make_supervisor(classify: Classifier, *, threshold: float) -> Any:
    """Build the supervisor node around an injected classifier."""

    def supervisor(state: MeshState) -> dict[str, Any]:
        try:
            classification = classify(state["query"])
        except Exception:  # noqa: BLE001 - any classifier failure fails closed
            # Fail closed, as retrieval does. A broken classifier must not pick a
            # specialist at random and let its answer be presented as grounded.
            return {
                "route": "refuse",
                "confidence": 0.0,
                "guard_flags": [*state["guard_flags"], "classifier_error"],
            }

        return {
            "route": decide_route(classification, threshold=threshold),
            # The gate changes the route, not the record of how unsure the model was.
            "confidence": classification.confidence,
        }

    return supervisor


def build_classifier(model: BaseChatModel) -> Classifier:
    """Wrap a chat model as a structured-output classifier."""
    structured = model.with_structured_output(Classification)

    def classify(query: str) -> Classification:
        result = structured.invoke(
            [("system", SYSTEM_PROMPT), ("human", query)],
        )
        return Classification.model_validate(result)

    return classify
