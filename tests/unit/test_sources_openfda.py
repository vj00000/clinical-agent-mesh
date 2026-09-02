"""Parsing openFDA label payloads.

YOUR BUILD. These tests are written and failing; make them pass one at a time,
top to bottom.

    make build-openfda

Tested against fixed payloads rather than the live API, so these assertions are
about payload handling and not about whatever api.fda.gov returned today.
"""

import pytest

# Excluded from the default run so `make check` stays green while
# parse_openfda_interactions is unimplemented. Run them with `make build-openfda`.
pytestmark = pytest.mark.todo

WITH_INTERACTIONS = """
{
  "results": [
    {
      "drug_interactions": [
        "Concomitant use with NSAIDs increases the risk of bleeding.",
        "Avoid   grapefruit   juice."
      ]
    }
  ]
}
"""

WITHOUT_SECTION = '{"results": [{"openfda": {"generic_name": ["metformin"]}}]}'

NO_RESULTS = '{"results": []}'


def _parse(payload: str) -> list[str]:
    """Imported inside the call, not at module scope.

    A module-level import of a function that does not exist yet fails at
    collection time, which takes the whole suite down with it rather than
    failing these tests alone -- and it happens before pytest can see the
    `todo` marker above.
    """
    from mesh.retrieval.sources import parse_openfda_interactions

    return parse_openfda_interactions(payload)


def test_every_interaction_section_becomes_a_note():
    notes = _parse(WITH_INTERACTIONS)

    assert len(notes) == 2
    assert notes[0].startswith("Concomitant use with NSAIDs")


def test_whitespace_is_normalised():
    notes = _parse(WITH_INTERACTIONS)

    assert notes[1] == "Avoid grapefruit juice."


def test_a_label_without_the_section_yields_nothing():
    """"No interactions listed" and "this label has no such section" are
    different claims. Only the second is true, so nothing is invented."""
    assert _parse(WITHOUT_SECTION) == []


def test_an_empty_result_set_yields_nothing():
    assert _parse(NO_RESULTS) == []
