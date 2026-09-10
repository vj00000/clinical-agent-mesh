"""Pairing medications against drug-label interaction text.

Not a model call, so it is fully testable: the note fetcher is injected, and
these assert on which pairs came back rather than on how a mock was used.
"""

from mesh.agents.discharge import build_interaction_lookup
from mesh.retrieval.sources import InteractionNote

NSAID_NOTE = InteractionNote(
    label_id="warf-1",
    text="Concomitant use with ibuprofen or other NSAIDs increases bleeding risk.",
)

CLASS_ONLY_NOTE = InteractionNote(
    label_id="warf-2",
    text="Many drugs interact with anticoagulants; review the full list.",
)


def _fetcher(notes: dict[str, list[InteractionNote]]):
    def fetch(drug: str) -> list[InteractionNote]:
        return notes.get(drug, [])

    return fetch


def test_a_note_naming_another_medication_becomes_an_interaction():
    lookup = build_interaction_lookup(_fetcher({"warfarin": [NSAID_NOTE]}))

    interactions = lookup(["warfarin", "ibuprofen"])

    assert len(interactions) == 1
    assert set(interactions[0].drugs) == {"warfarin", "ibuprofen"}


def test_a_note_that_names_no_other_medication_is_ignored():
    """Label interaction sections warn about whole drug classes. Reporting those
    would warn about every drug on earth and bury the pair that matters."""
    lookup = build_interaction_lookup(_fetcher({"warfarin": [CLASS_ONLY_NOTE]}))

    assert lookup(["warfarin", "metformin"]) == []


def test_the_interaction_carries_the_label_it_came_from():
    """It joins retrieved_ids and gets cited, so without provenance guard_out
    would reject the whole answer."""
    lookup = build_interaction_lookup(_fetcher({"warfarin": [NSAID_NOTE]}))

    interaction = lookup(["warfarin", "ibuprofen"])[0]

    assert interaction.chunk_id == "openfda:warf-1"
    assert interaction.source == "openfda"


def test_a_pair_mentioned_by_both_labels_is_reported_once():
    """Only one of the two labels may mention the other, so both directions are
    checked -- which means a mutual mention would otherwise duplicate."""
    shared = InteractionNote(label_id="shared-1", text="warfarin and ibuprofen together bleed.")
    lookup = build_interaction_lookup(_fetcher({"warfarin": [shared], "ibuprofen": [shared]}))

    assert len(lookup(["warfarin", "ibuprofen"])) == 1


def test_a_drug_is_never_paired_with_itself():
    lookup = build_interaction_lookup(
        _fetcher({"warfarin": [InteractionNote(label_id="w", text="warfarin warfarin warfarin")]})
    )

    assert lookup(["warfarin", "metformin"]) == []


def test_matching_ignores_case():
    note = InteractionNote(label_id="w", text="Avoid IBUPROFEN while taking this.")
    lookup = build_interaction_lookup(_fetcher({"warfarin": [note]}))

    assert len(lookup(["warfarin", "ibuprofen"])) == 1


def test_a_drug_with_no_labels_contributes_nothing():
    lookup = build_interaction_lookup(_fetcher({}))

    assert lookup(["warfarin", "ibuprofen"]) == []
