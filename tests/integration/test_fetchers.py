"""Fetchers against the live public APIs.

Marked `network` and excluded from the default run: NCBI rate-limits keyless
callers, so CI must not depend on it. Run with `make test-network`.
"""

import httpx
import pytest

from mesh.retrieval.sources import fetch_medlineplus, fetch_pubmed

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
