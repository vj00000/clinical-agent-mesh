"""The parent/child state contract.

guard_out can only verify grounding if every citation points at a real
retrieved chunk, so an empty chunk_id must be impossible to construct.
"""

import pytest
from pydantic import ValidationError

from mesh.state import Citation, new_state


def test_citation_without_a_chunk_id_is_rejected():
    with pytest.raises(ValidationError):
        Citation(chunk_id="", source="cdc-hypertension-2024", quote="First-line therapy is...")


def test_new_state_starts_with_no_route_and_no_citations():
    state = new_state("what is first-line therapy for hypertension?")

    assert state["query"] == "what is first-line therapy for hypertension?"
    assert state["route"] is None
    assert state["citations"] == []
    assert state["confidence"] == 0.0


def test_each_new_state_gets_its_own_trace_id():
    first = new_state("query one")
    second = new_state("query two")

    assert first["trace_id"] != second["trace_id"]
