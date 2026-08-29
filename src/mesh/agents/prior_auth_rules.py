"""Deterministic coverage decisions for the prior-auth specialist.

The model reads the policy and marks each criterion against the request. It never
decides the outcome: that is computed here, by a rule a reviewer can read in ten
seconds and a test can pin exactly.

Demo depth. Real prior authorisation runs on payer-specific policy engines with
appeal paths and human review; this reads public CMS coverage determinations and
demonstrates the shape.
"""

from enum import StrEnum

from pydantic import BaseModel, Field


class Decision(StrEnum):
    APPROVE = "approve"
    DENY = "deny"
    MORE_INFO = "more_info"


class Criterion(BaseModel):
    """One requirement from a coverage policy, marked against the request."""

    text: str = Field(min_length=1)

    # None means the request is silent on this requirement: unknown, not failed.
    # Collapsing the two is how a missing form becomes a denial.
    met: bool | None

    chunk_id: str = Field(min_length=1)


def decide_coverage(criteria: list[Criterion]) -> Decision:
    """Approve only when every criterion is met.

    The order of these checks is the policy. A criterion that is definitively
    unmet denies the request even when others are unresolved: gathering more
    information cannot rescue a requirement that has already failed. No criteria
    at all is not an approval either -- it means no policy was retrieved to judge
    against.
    """
    if not criteria:
        return Decision.MORE_INFO

    if any(criterion.met is False for criterion in criteria):
        return Decision.DENY

    if any(criterion.met is None for criterion in criteria):
        return Decision.MORE_INFO

    return Decision.APPROVE


_HEADLINE = {
    Decision.APPROVE: "Every criterion in the retrieved coverage policy is met.",
    Decision.DENY: (
        "The retrieved coverage policy has at least one criterion this request does not meet."
    ),
    Decision.MORE_INFO: (
        "I can't decide this yet: the policy has criteria the request does not resolve."
    ),
}

_MARK: dict[bool | None, str] = {
    True: "met",
    False: "not met",
    None: "unresolved",
}


def render_decision(decision: Decision, criteria: list[Criterion]) -> str:
    """Compose the answer from the decision and the criteria it rests on.

    Written in code rather than by the model so the prose cannot contradict the
    verdict. An approval that reads like a denial is worse than either.
    """
    lines = [_HEADLINE[decision]]

    if criteria:
        lines.append("")
        lines.extend(f"- [{_MARK[criterion.met]}] {criterion.text}" for criterion in criteria)

    return "\n".join(lines)
