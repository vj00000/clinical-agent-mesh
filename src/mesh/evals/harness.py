"""The evaluation harness: run the golden set through the mesh and score it.

Running costs one classifier call plus the chosen route's calls per case, so
this is not part of `make check`. `make eval` runs it, prints the table, and
exits non-zero when a metric falls below its threshold -- which is what makes it
a regression gate rather than a report nobody reads.

The mesh is injected, so the harness itself is testable with a stub model.
"""

import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from mesh.evals.metrics import (
    GoldenCase,
    RunOutcome,
    answer_rate,
    citation_accuracy,
    refusal_correctness,
    routing_accuracy,
)
from mesh.graph import OUT_OF_SCOPE_REFUSAL, RETRIEVER_REFUSAL
from mesh.guardrails.nodes import UNGROUNDED_REFUSAL
from mesh.state import new_state

GOLDEN_PATH = Path(__file__).resolve().parents[3] / "evals" / "golden" / "mesh_cases.json"

# Every way the system can decline to answer. Collected here rather than
# checked by substring: a refusal is an exact known string, and matching
# loosely would silently reclassify a real answer that happened to contain
# "I can't".
REFUSAL_TEXTS = frozenset({OUT_OF_SCOPE_REFUSAL, RETRIEVER_REFUSAL, UNGROUNDED_REFUSAL})

# Gate thresholds. Deliberately not aspirational: a gate set above what the
# system currently does is a gate someone disables on their second red build.
# Raise them as the numbers improve.
THRESHOLDS = {
    "refusal_correctness": 0.70,
    "citation_accuracy": 0.95,
    "routing_accuracy": 0.80,
}


class EvalReport(BaseModel):
    cases: int
    refusal_correctness: float
    citation_accuracy: float
    routing_accuracy: float
    answer_rate: float
    answered_when_unanswerable: int
    refused_when_answerable: int

    def failures(self) -> list[str]:
        """Which thresholds this run misses, if any."""
        scores = {
            "refusal_correctness": self.refusal_correctness,
            "citation_accuracy": self.citation_accuracy,
            "routing_accuracy": self.routing_accuracy,
        }

        return [
            f"{name}: {scores[name]:.3f} below threshold {floor:.2f}"
            for name, floor in THRESHOLDS.items()
            if scores[name] < floor
        ]


def load_golden_cases(path: Path | None = None) -> list[GoldenCase]:
    source = path or GOLDEN_PATH

    return [GoldenCase.model_validate(case) for case in json.loads(source.read_text())]


def is_refusal(answer: str, route: str | None) -> bool:
    """Whether the system declined to answer.

    A clarifying question counts. It is not a failure -- asking beats guessing --
    but it is not an answer either, and scoring it as one would let a system
    that asks for clarification every time look perfectly calibrated.
    """
    return route in {"refuse", "clarify"} or answer in REFUSAL_TEXTS


def run_cases(mesh: Any, cases: Sequence[GoldenCase]) -> list[RunOutcome]:
    """Put every case through the mesh and record what happened.

    A case that raises is recorded as a refusal with an error route rather than
    aborting the run: one bad case must not cost the whole evaluation, which on
    a paid API is real money.
    """
    outcomes: list[RunOutcome] = []

    for case in cases:
        try:
            result = mesh.invoke(new_state(case.question))
        except Exception as error:  # noqa: BLE001 - a failed case is a miss, not a crash
            outcomes.append(
                RunOutcome(
                    case=case,
                    route="__error__",
                    answer=f"{type(error).__name__}: {error}",
                    citations=[],
                    retrieved_ids=[],
                    refused=True,
                )
            )
            continue

        outcomes.append(
            RunOutcome(
                case=case,
                route=result["route"] or "__none__",
                answer=result["answer"],
                citations=result["citations"],
                retrieved_ids=result["retrieved_ids"],
                refused=is_refusal(result["answer"], result["route"]),
            )
        )

    return outcomes


def score(outcomes: Sequence[RunOutcome]) -> EvalReport:
    refusal = refusal_correctness(outcomes)

    return EvalReport(
        cases=len(outcomes),
        refusal_correctness=refusal.accuracy,
        citation_accuracy=citation_accuracy(outcomes),
        routing_accuracy=routing_accuracy(outcomes),
        answer_rate=answer_rate(outcomes),
        answered_when_unanswerable=refusal.answered_when_unanswerable,
        refused_when_answerable=refusal.refused_when_answerable,
    )


def format_report(report: EvalReport) -> str:
    """The metrics table, in the shape the README quotes."""
    lines = [
        "",
        f"Cases: {report.cases}",
        "",
        "| Metric | Score | Threshold |",
        "|---|---|---|",
    ]
    for name, floor in THRESHOLDS.items():
        value = getattr(report, name)
        lines.append(f"| {name.replace('_', ' ')} | {value:.3f} | {floor:.2f} |")

    lines.extend(
        [
            f"| answer rate | {report.answer_rate:.3f} | - |",
            "",
            f"Answered when unanswerable: {report.answered_when_unanswerable}  "
            "(the dangerous direction)",
            f"Refused when answerable:    {report.refused_when_answerable}  "
            "(annoying, not dangerous)",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    """Entry point for `make eval`. Costs one run of the mesh per case."""
    import httpx

    from mesh.composition import build_default_mesh
    from mesh.models.config import Settings
    from mesh.models.providers import build_chat_model, build_embeddings
    from mesh.retrieval.sources import InteractionNote, fetch_openfda_interaction_notes

    try:
        settings = Settings()
    except ValidationError:
        print(
            "OPENAI_API_KEY is not set. Run `cp .env.example .env` and add your key.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None

    cases = load_golden_cases()
    print(f"Running {len(cases)} cases against {settings.chat_model}...", flush=True)

    with httpx.Client(timeout=30.0) as client:

        def fetch_notes(drug: str) -> list[InteractionNote]:
            return fetch_openfda_interaction_notes(drug, limit=1, client=client)

        mesh = build_default_mesh(
            settings=settings,
            model=build_chat_model(settings),
            embeddings=build_embeddings(settings),
            fetch_notes=fetch_notes,
        )
        outcomes = run_cases(mesh, cases)

    report = score(outcomes)
    print(format_report(report))

    failures = report.failures()
    if failures:
        print("\nFAILED:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        raise SystemExit(1)

    print("\nAll thresholds met.")


if __name__ == "__main__":
    main()
