"""Prompt-injection detection for guard_in and for retrieved chunks.

Each pattern requires *two* elements together — an overriding verb plus a
reference to prior instructions — rather than either alone. Matching the word
"instructions" by itself would flag "patient instructions" and "discharge
instructions", which are ordinary clinical phrases, and a guardrail that fires
on normal domain vocabulary gets switched off.

These are heuristics, not a proof. They raise the cost of the obvious attacks;
they do not make injection impossible.
"""

import re

from pydantic import BaseModel

# `[\s\S]{0,N}?` rather than `.{0,N}?` so a match can span the line breaks that
# poisoned corpus chunks tend to contain.
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "instruction_override",
        re.compile(
            r"\b(?:ignore|disregard|forget|override|bypass)\b"
            r"[\s\S]{0,20}?\b(?:previous|prior|above|preceding|earlier|all|your|these|those)\b"
            r"[\s\S]{0,20}?\b(?:instruction|rule|prompt|direction|guideline|constraint)s?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "prompt_disclosure",
        re.compile(
            r"\b(?:reveal|repeat|print|show|output|display|disclose)\b"
            r"[\s\S]{0,30}?\b(?:system\s+prompt|initial\s+prompt|"
            r"your\s+(?:instructions|prompt|rules))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "role_marker",
        re.compile(r"(?:^|\n)\s*(?:system|assistant|user)\s*:", re.IGNORECASE),
    ),
    (
        # Persona jailbreaks: reassign who the assistant is, then assert the new
        # persona has no limits. Both halves are required, as everywhere else
        # here -- "act as a nurse would" is an ordinary clinical framing and must
        # not fire.
        #
        # The residual false positive is a second-person sentence like "you are
        # now on metformin with no restrictions on diet". guard_in scans only
        # what the user typed, and a patient does not write that about
        # themselves, so the trade is accepted.
        "persona_override",
        re.compile(
            r"\b(?:you\s+are\s+now|from\s+now\s+on\s+you\s+are|"
            r"pretend\s+(?:to\s+be|you\s+are)|act\s+as|roleplay\s+as)\b"
            r"[\s\S]{0,60}?"
            r"\b(?:DAN|jailbroken|unfiltered|unrestricted|"
            r"no\s+(?:restrictions|rules|limits|filters|guardrails))\b",
            re.IGNORECASE,
        ),
    ),
)


class InjectionVerdict(BaseModel):
    suspicious: bool
    reasons: list[str]


def detect_injection(text: str) -> InjectionVerdict:
    """Report which injection patterns, if any, the text matches."""
    reasons = [label for label, pattern in _PATTERNS if pattern.search(text)]

    return InjectionVerdict(suspicious=bool(reasons), reasons=reasons)
