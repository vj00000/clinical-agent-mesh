"""PHI redaction for the guard_in node.

Deliberately conservative. Every pattern here is anchored on structure that
clinical values do not share, because over-redaction is the worse failure: a
mangled blood pressure or dosage changes the question the model is answering,
and it does so silently.

Known limitation: patient *names* are not detected. Regex cannot distinguish a
surname from a drug name or an eponymous condition, and a name-matching pass
would redact "Crohn" and "Parkinson". Name handling belongs to a dedicated NER
model, which is out of scope here and is stated rather than pretended.
"""

import re

from pydantic import BaseModel

# A full date needs three components. Two-component values like 120/80 are
# clinical readings and must survive.
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("MRN", re.compile(r"\bMRN\s*[:#]?\s*\d{4,}\b", re.IGNORECASE)),
    ("PHONE", re.compile(r"\b(?:\+?\d{1,2}[\s.-])?\d{3}[\s.-]\d{3}[\s.-]\d{4}\b")),
    ("DATE", re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")),
)


class RedactionResult(BaseModel):
    text: str
    found: list[str]


def redact_phi(text: str) -> RedactionResult:
    """Replace identifiers with category placeholders.

    Idempotent: placeholders contain no digits or at-signs, so a second pass over
    already-redacted text matches nothing and changes nothing.
    """
    redacted = text
    found: list[str] = []

    for label, pattern in _PATTERNS:
        redacted, substitutions = pattern.subn(f"[{label}]", redacted)
        if substitutions:
            found.append(label)

    return RedactionResult(text=redacted, found=found)
