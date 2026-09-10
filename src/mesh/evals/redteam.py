"""The adversarial suite.

Split by what each case needs. Injection and PHI cases are settled entirely by
`guard_in`, which contains no model call, so those run in the ordinary test
suite with no key and no network -- the safety properties most worth pinning are
the ones that cost nothing to check.

The remaining categories need the whole mesh and a real model, and are scored by
the harness.
"""

import json
from pathlib import Path

from pydantic import BaseModel, Field

from mesh.guardrails.injection import detect_injection
from mesh.guardrails.phi import redact_phi

# Repo-relative so the suite travels with the code and CI needs no fixture path.
REDTEAM_PATH = Path(__file__).resolve().parents[3] / "evals" / "redteam" / "cases.json"

# Categories decidable without a model.
DETERMINISTIC_CATEGORIES = frozenset({"injection", "phi"})


class RedTeamCase(BaseModel):
    query: str = Field(min_length=1)
    expect: str = Field(min_length=1)
    category: str = Field(min_length=1)


def load_redteam_cases(path: Path | None = None) -> list[RedTeamCase]:
    source = path or REDTEAM_PATH

    return [RedTeamCase.model_validate(case) for case in json.loads(source.read_text())]


def blocks_injection(query: str) -> bool:
    """Whether guard_in would stop this query before the supervisor sees it."""
    return detect_injection(redact_phi(query).text).suspicious


def redacts_identifiers(query: str) -> bool:
    """Whether guard_in strips at least one identifier from this query."""
    return bool(redact_phi(query).found)
