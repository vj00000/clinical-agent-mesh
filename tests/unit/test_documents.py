"""Document normalisation.

MedlinePlus returns HTML fragments inside its summaries and PubMed abstracts
arrive with hard line wrapping, so raw payloads cannot go straight into chunks:
markup would be embedded as if it were clinical text.
"""

import pytest
from pydantic import ValidationError

from mesh.retrieval.documents import Document, clean_text


def test_html_tags_are_stripped_but_their_text_is_kept():
    assert clean_text("<p>Lisinopril is an <b>ACE</b> inhibitor.</p>") == (
        "Lisinopril is an ACE inhibitor."
    )


def test_html_entities_are_decoded():
    assert clean_text("blood pressure &lt; 130&#47;80 &amp; falling") == (
        "blood pressure < 130/80 & falling"
    )


def test_hard_line_wrapping_and_repeated_spaces_collapse_to_single_spaces():
    assert clean_text("first-line\n  therapy\n\nfor   hypertension") == (
        "first-line therapy for hypertension"
    )


def test_entity_encoded_markup_is_stripped_too():
    """MedlinePlus double-encodes: its FullSummary contains &lt;p&gt;, not <p>.

    Unescaping without a second strip pass would leave literal tags in the text
    and embed markup as if it were clinical prose.
    """
    raw = "What is blood pressure?&lt;p&gt;It is the force of your blood.&lt;/p&gt;"

    assert clean_text(raw) == "What is blood pressure? It is the force of your blood."


def test_plain_text_is_left_alone():
    assert clean_text("Metformin is first-line therapy.") == "Metformin is first-line therapy."


def test_a_document_without_text_is_rejected():
    with pytest.raises(ValidationError):
        Document(doc_id="pmid:1", source="pubmed", title="A trial", text="")


def test_a_document_without_a_source_is_rejected():
    with pytest.raises(ValidationError):
        Document(doc_id="pmid:1", source="", title="A trial", text="some text")
