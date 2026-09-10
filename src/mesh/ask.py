"""Put one question through the mesh and print the answer. Entry point for `make ask`.

The smallest thing that proves the system works end to end: real corpora, real
retrieval, real guardrails, one real model call per node on the chosen route.
"""

import sys

import httpx
from pydantic import ValidationError

from mesh.composition import build_default_mesh
from mesh.models.config import Settings
from mesh.models.providers import build_chat_model, build_embeddings
from mesh.retrieval.sources import InteractionNote, fetch_openfda_interaction_notes
from mesh.state import new_state

_HTTP_TIMEOUT = 30.0


def main() -> None:
    query = " ".join(sys.argv[1:]).strip()
    if not query:
        print('Usage: make ask q="what is first-line therapy for hypertension?"', file=sys.stderr)
        raise SystemExit(2)

    try:
        settings = Settings()
    except ValidationError:
        print(
            "OPENAI_API_KEY is not set. Run `cp .env.example .env` and add your key.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None

    # The client outlives the call because the interaction lookup runs mid-graph,
    # not during construction.
    with httpx.Client(timeout=_HTTP_TIMEOUT) as client:

        def fetch_notes(drug: str) -> list[InteractionNote]:
            return fetch_openfda_interaction_notes(drug, limit=1, client=client)

        mesh = build_default_mesh(
            settings=settings,
            model=build_chat_model(settings),
            embeddings=build_embeddings(settings),
            fetch_notes=fetch_notes,
        )

        result = mesh.invoke(new_state(query))

    print(f"\nroute: {result['route']}  (confidence {result['confidence']:.2f})")
    if result["guard_flags"]:
        print(f"guards: {', '.join(result['guard_flags'])}")

    print(f"\n{result['answer']}\n")

    if result["citations"]:
        print("sources:")
        for citation in result["citations"]:
            print(f"  [{citation.chunk_id}] {citation.source}")
    else:
        # Not an error: refusals and clarifying questions cite nothing by design.
        print("(no citations)")


if __name__ == "__main__":
    main()
