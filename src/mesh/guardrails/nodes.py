"""Guardrails as graph nodes.

Safety runs as its own step with its own tests rather than as a paragraph
appended to a prompt, so it cannot be argued away by the model and its decisions
appear in the state's audit trail.
"""

from typing import Any

from mesh.guardrails.citations import verify_citations
from mesh.guardrails.injection import detect_injection
from mesh.guardrails.phi import redact_phi
from mesh.state import MeshState

UNGROUNDED_REFUSAL = (
    "I can't answer that from the sources I retrieved. Rather than give you an "
    "unsupported clinical claim, I'd rather tell you I don't have the evidence."
)


def guard_in(state: MeshState) -> dict[str, Any]:
    """Redact identifiers and block injection before the supervisor sees the query.

    Redaction happens first: an injection attempt hidden in text that also
    contains PHI must still be scanned, and the scan should run on what the rest
    of the graph will actually receive.
    """
    redaction = redact_phi(state["query"])
    flags = [f"phi:{category}" for category in redaction.found]

    injection = detect_injection(redaction.text)
    flags.extend(f"injection:{reason}" for reason in injection.reasons)

    return {
        "query": redaction.text,
        "route": "refuse" if injection.suspicious else None,
        "guard_flags": flags,
    }


# Neither a refusal nor a clarifying question asserts anything clinical, so
# neither has anything to cite. Demanding citations from them would replace a
# perfectly good question with a refusal.
_UNCITED_ROUTES = frozenset({"refuse", "clarify"})

# A red-flag escalation is deterministic: the rules fired, the model did not
# assert it. It is exempt for the same reason a refusal is — it makes no claim
# about the evidence — and it is a flag rather than a route so the supervisor's
# routing vocabulary stays unchanged.
#
# Without this, a retrieval outage silently converts "call an ambulance" into
# "I don't have the evidence", which is the worst output this system can produce.
SAFETY_ESCALATION_FLAG = "safety_escalation"


def guard_out(state: MeshState) -> dict[str, Any]:
    """Refuse rather than emit a clinical claim that cites nothing real."""
    if state["route"] in _UNCITED_ROUTES or SAFETY_ESCALATION_FLAG in state["guard_flags"]:
        return {"answer": state["answer"], "citations": state["citations"]}

    verdict = verify_citations(state["citations"], retrieved=set(state["retrieved_ids"]))
    if verdict.grounded:
        return {"answer": state["answer"], "citations": state["citations"]}

    flags = list(state["guard_flags"])
    flags.extend(f"ungrounded:{chunk_id}" for chunk_id in verdict.unsupported)
    if not state["citations"]:
        flags.append("ungrounded:no_citations")

    return {"answer": UNGROUNDED_REFUSAL, "citations": [], "guard_flags": flags}
