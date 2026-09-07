"""Parsing openFDA label payloads.

Tested against fixed payloads rather than the live API, so these assertions are
about payload handling and not about whatever api.fda.gov returned today.
"""

from mesh.retrieval.sources import parse_openfda_interactions, parse_openfda_labels

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


def test_every_interaction_section_becomes_a_note():
    notes = parse_openfda_interactions(WITH_INTERACTIONS)

    assert len(notes) == 2
    assert notes[0].startswith("Concomitant use with NSAIDs")


def test_whitespace_is_normalised():
    notes = parse_openfda_interactions(WITH_INTERACTIONS)

    assert notes[1] == "Avoid grapefruit juice."


def test_a_label_without_the_section_yields_nothing():
    """ "No interactions listed" and "this label has no such section" are
    different claims. Only the second is true, so nothing is invented."""
    assert parse_openfda_interactions(WITHOUT_SECTION) == []


def test_an_empty_result_set_yields_nothing():
    assert parse_openfda_interactions(NO_RESULTS) == []


LABEL = """
{
  "results": [
    {
      "id": "abc-123",
      "drug_interactions": ["Avoid NSAIDs."],
      "dosage_and_administration": ["Take 5mg once daily."],
      "spl_unclassified_section": ["prescriber-only boilerplate"]
    }
  ]
}
"""


def test_a_label_becomes_one_document():
    documents = parse_openfda_labels(LABEL, drug="warfarin")

    assert len(documents) == 1
    assert documents[0].doc_id == "openfda:abc-123"
    assert documents[0].source == "openfda"


def test_only_the_sections_a_patient_needs_are_kept():
    documents = parse_openfda_labels(LABEL, drug="warfarin")

    assert "Take 5mg once daily." in documents[0].text
    assert "prescriber-only boilerplate" not in documents[0].text


def test_a_label_with_none_of_those_sections_is_dropped():
    """No usable text means nothing to ground an answer on."""
    payload = '{"results": [{"id": "x", "spl_unclassified_section": ["blah"]}]}'

    assert parse_openfda_labels(payload, drug="warfarin") == []


def test_a_label_without_an_id_falls_back_to_the_drug_name():
    """Chunk ids are a hash of source plus text, so an unstable doc_id would
    invalidate citations on every re-ingest."""
    payload = '{"results": [{"drug_interactions": ["Avoid NSAIDs."]}]}'

    documents = parse_openfda_labels(payload, drug="warfarin")

    assert documents[0].doc_id == "openfda:warfarin-0"
