"""Parsing the CMS coverage policy index.

Tested against a fixed payload rather than the live API, so these assertions are
about payload handling and not about what CMS published today.
"""

from mesh.retrieval.sources import parse_cms_coverage

REPORT = """
{
  "meta": {"status": {"id": 200}},
  "data": [
    {
      "document_id": 38568,
      "document_display_id": "L38568",
      "title": "MolDX: Molecular Testing for Solid Organ Allograft Rejection",
      "contractor_name_type": "Palmetto GBA\\r\\n(MAC - Part A)",
      "effective_date": "07/10/2025",
      "note": "Retired"
    },
    {
      "document_id": 90210,
      "document_display_id": "",
      "title": "A policy with no display id"
    }
  ]
}
"""

NO_DATA = '{"meta": {"status": {"id": 200}}, "data": []}'


def test_a_policy_becomes_one_document():
    documents = parse_cms_coverage(REPORT, document_type="LCD")

    assert len(documents) == 1
    assert documents[0].doc_id == "cms:L38568"
    assert documents[0].source == "cms-coverage"


def test_the_text_carries_what_the_policy_is_and_who_issued_it():
    document = parse_cms_coverage(REPORT, document_type="LCD")[0]

    assert "LCD L38568: MolDX" in document.text
    assert "contractor: Palmetto GBA (MAC - Part A)" in document.text
    assert "effective date: 07/10/2025" in document.text


def test_a_record_without_an_identifier_is_dropped():
    """The id is the citation. A policy that cannot be cited cannot ground an
    answer, so it is not admitted to the corpus."""
    documents = parse_cms_coverage(REPORT, document_type="LCD")

    assert [d.doc_id for d in documents] == ["cms:L38568"]


def test_an_empty_report_yields_nothing():
    assert parse_cms_coverage(NO_DATA, document_type="NCD") == []


def test_the_document_type_appears_in_the_text():
    """NCDs and LCDs answer different questions; the retriever should be able to
    tell them apart lexically."""
    document = parse_cms_coverage(REPORT, document_type="NCD")[0]

    assert document.text.startswith("NCD L38568:")
