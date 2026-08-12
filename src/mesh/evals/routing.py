"""Scoring for the routing benchmark.

Accuracy is the headline, but the confusion matrix is what you act on: it names
the pair of specialists the classifier confuses, which is the prompt to fix.
"""

import json
from collections.abc import Callable, Sequence
from pathlib import Path

from pydantic import BaseModel

from mesh.agents.routing import Classification

Classifier = Callable[[str], Classification]

CLASSIFIER_ERROR = "__error__"

# Repo-relative so the benchmark travels with the code and CI needs no fixture path.
BENCHMARK_PATH = Path(__file__).resolve().parents[3] / "evals" / "golden" / "routing_cases.json"


class RoutingCase(BaseModel):
    query: str
    expected: str


class Misroute(BaseModel):
    query: str
    expected: str
    predicted: str


class RoutingReport(BaseModel):
    total: int
    correct: int
    accuracy: float
    misroutes: list[Misroute]
    # (expected, predicted) -> count
    confusion: dict[tuple[str, str], int]


def load_routing_cases(path: Path | None = None) -> list[RoutingCase]:
    """Load the labelled routing benchmark."""
    source = path or BENCHMARK_PATH
    return [RoutingCase.model_validate(case) for case in json.loads(source.read_text())]


def score_routing(cases: Sequence[RoutingCase], classify: Classifier) -> RoutingReport:
    """Run every case through the classifier and score the results.

    A classifier error scores as a miss rather than aborting: one bad response
    must not cost the whole benchmark run, which on a paid API is real money.
    """
    if not cases:
        raise ValueError("cannot score an empty case set: the result would be meaningless")

    correct = 0
    misroutes: list[Misroute] = []
    confusion: dict[tuple[str, str], int] = {}

    for case in cases:
        # Annotated as str, not Route: a failed call records a sentinel that is
        # deliberately outside the route vocabulary.
        predicted: str
        try:
            predicted = classify(case.query).route
        except Exception:  # noqa: BLE001 - a failed call is a miss, not a crash
            predicted = CLASSIFIER_ERROR

        key = (case.expected, predicted)
        confusion[key] = confusion.get(key, 0) + 1

        if predicted == case.expected:
            correct += 1
        else:
            misroutes.append(
                Misroute(query=case.query, expected=case.expected, predicted=predicted)
            )

    return RoutingReport(
        total=len(cases),
        correct=correct,
        accuracy=correct / len(cases),
        misroutes=misroutes,
        confusion=confusion,
    )


def main() -> None:
    """Entry point for `make eval-routing`. Costs one classifier call per case."""
    import sys

    from pydantic import ValidationError

    from mesh.agents.supervisor import build_classifier
    from mesh.models.config import Settings
    from mesh.models.providers import build_chat_model

    try:
        settings = Settings()
    except ValidationError:
        print(
            "OPENAI_API_KEY is not set. Run `cp .env.example .env` and add your key.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None

    cases = load_routing_cases()
    print(f"Scoring {len(cases)} routing cases against {settings.chat_model}...")

    report = score_routing(cases, build_classifier(build_chat_model(settings)))

    print(f"\nAccuracy: {report.accuracy:.1%}  ({report.correct}/{report.total})")

    if report.misroutes:
        print("\nMisroutes:")
        for miss in report.misroutes:
            print(f"  [{miss.expected} -> {miss.predicted}] {miss.query}")

        pairs = sorted(
            ((k, v) for k, v in report.confusion.items() if k[0] != k[1]),
            key=lambda item: -item[1],
        )
        print("\nMost confused pairs:")
        for (expected, predicted), count in pairs[:5]:
            print(f"  {expected} mistaken for {predicted}: {count}")


if __name__ == "__main__":
    main()
