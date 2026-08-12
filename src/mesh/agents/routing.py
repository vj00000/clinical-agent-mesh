"""Supervisor routing policy.

The LLM proposes a route; this decides whether to trust it. Keeping the policy
out of the prompt makes it testable and auditable.
"""

from pydantic import BaseModel, Field

from mesh.state import Route

SPECIALISTS: frozenset[str] = frozenset({"guideline", "triage", "prior_auth", "discharge"})


class Classification(BaseModel):
    """What the supervisor LLM is asked to return."""

    route: Route
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


def decide_route(classification: Classification, *, threshold: float) -> Route:
    """Accept the proposed route, ask for clarification, or refuse.

    A refusal is honoured regardless of confidence: an out-of-scope question does
    not become in-scope because the classifier hedged about it.
    """
    if classification.route not in SPECIALISTS:
        return classification.route

    if classification.confidence < threshold:
        return "clarify"

    return classification.route
