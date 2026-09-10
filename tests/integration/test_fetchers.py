"""Fetchers against the live public APIs.

Marked `network` and excluded from the default run: NCBI rate-limits keyless
callers, so CI must not depend on it. Run with `make test-network`.
"""

import httpx
import pytest

from mesh.retrieval.sources import (
    fetch_cms_coverage,
    fetch_medlineplus,
    fetch_openfda_labels,
    fetch_pubmed,
)

pytestmark = pytest.mark.network


@pytest.fixture
def client():
    with httpx.Client(timeout=30.0) as c:
        yield c


def test_pubmed_returns_documents_with_grounded_text(client):
    docs = fetch_pubmed("hypertension guideline", limit=3, client=client)

    assert docs, "expected at least one PubMed hit for a common clinical query"
    assert all(d.doc_id.startswith("pmid:") for d in docs)
    assert all(d.text.strip() for d in docs)
    assert all(d.source == "pubmed" for d in docs)


def test_pubmed_respects_the_requested_limit(client):
    docs = fetch_pubmed("diabetes management", limit=2, client=client)

    assert len(docs) <= 2


def test_medlineplus_returns_patient_facing_topics(client):
    docs = fetch_medlineplus("hypertension", limit=2, client=client)

    assert docs
    assert all(d.source == "medlineplus" for d in docs)
    assert all(d.text.strip() for d in docs)


def test_fetched_text_carries_no_residual_markup(client):
    """Regression guard: MedlinePlus double-encodes its summaries."""
    docs = fetch_medlineplus("hypertension", limit=2, client=client)

    for doc in docs:
        assert "<" not in doc.text
        assert "&lt;" not in doc.text


def test_openfda_returns_label_documents(client):
    documents = fetch_openfda_labels("warfarin", limit=1, client=client)

    assert documents
    assert documents[0].source == "openfda"
    assert documents[0].text.strip()


def test_an_unknown_drug_yields_nothing_rather_than_raising(client):
    """openFDA answers 404 for a search that matches nothing. A drug with no
    label on file is not an error."""
    assert fetch_openfda_labels("notarealdrugname", limit=1, client=client) == []


def test_cms_coverage_returns_both_ncds_and_lcds(client):
    documents = fetch_cms_coverage(client, limit=3)

    kinds = {document.text.split()[0] for document in documents}
    assert kinds == {"NCD", "LCD"}


def test_every_cms_document_is_citable(client):
    """The document id is the citation. One without an id could not be cited."""
    documents = fetch_cms_coverage(client, limit=3)

    assert all(document.doc_id.startswith("cms:") for document in documents)
