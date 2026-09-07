"""Parsers for the public clinical corpora.

Parsing is separated from fetching so payload handling is tested against fixed
payloads rather than against whatever the API returned today.
"""

import json
from xml.etree import ElementTree

import httpx

from mesh.retrieval.documents import Document, clean_text

PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
MEDLINEPLUS_URL = "https://wsearch.nlm.nih.gov/ws/query"
OPENFDA_LABEL_URL = "https://api.fda.gov/drug/label.json"

# Label sections a discharge summary actually draws on. A full SPL label runs to
# dozens of fields written for prescribers; embedding all of it buries the four
# that answer "how do I take this, and what should I watch for".
_LABEL_SECTIONS = (
    "drug_interactions",
    "warnings_and_cautions",
    "dosage_and_administration",
    "patient_medication_information",
)


def fetch_pubmed(query: str, *, limit: int, client: httpx.Client) -> list[Document]:
    """Search PubMed, then fetch the abstracts for the hits.

    Two calls are unavoidable: esearch returns ids only, efetch returns text.
    """
    search = client.get(
        f"{PUBMED_BASE}/esearch.fcgi",
        params={"db": "pubmed", "term": query, "retmax": limit, "retmode": "json"},
    )
    search.raise_for_status()

    ids = parse_esearch_ids(search.text)
    if not ids:
        return []

    articles = client.get(
        f"{PUBMED_BASE}/efetch.fcgi",
        params={"db": "pubmed", "id": ",".join(ids), "retmode": "xml"},
    )
    articles.raise_for_status()

    return parse_pubmed_articles(articles.text)


def fetch_medlineplus(term: str, *, limit: int, client: httpx.Client) -> list[Document]:
    """Search MedlinePlus health topics — patient-facing, public domain."""
    response = client.get(
        MEDLINEPLUS_URL,
        params={"db": "healthTopics", "term": term, "retmax": limit},
    )
    response.raise_for_status()

    return parse_medlineplus_topics(response.text)


def parse_esearch_ids(payload: str) -> list[str]:
    """Pull the PMID list out of an esearch JSON response."""
    result = json.loads(payload).get("esearchresult", {})
    return list(result.get("idlist", []))


def _content(document: ElementTree.Element, name: str) -> str | None:
    for node in document.findall("content"):
        if node.get("name") == name:
            return clean_text("".join(node.itertext()))
    return None


def parse_medlineplus_topics(xml: str) -> list[Document]:
    """Turn a wsearch result set into documents, one per health topic.

    Topics without a FullSummary are dropped for the same reason abstract-less
    PubMed records are: there is no text to ground an answer on.
    """
    root = ElementTree.fromstring(xml)
    documents: list[Document] = []

    for node in root.iter("document"):
        url = node.get("url")
        summary = _content(node, "FullSummary")
        if not url or not summary:
            continue

        documents.append(
            Document(
                doc_id=url,
                source="medlineplus",
                title=_content(node, "title") or "",
                text=summary,
            )
        )

    return documents


def parse_pubmed_articles(xml: str) -> list[Document]:
    """Turn an efetch PubmedArticleSet into documents, one per article.

    Articles without an abstract are dropped: a title alone cannot ground an
    answer, and admitting it would let the retriever cite an empty source.
    """
    root = ElementTree.fromstring(xml)
    documents: list[Document] = []

    for citation in root.iter("MedlineCitation"):
        pmid_node = citation.find("PMID")
        title_node = citation.find("./Article/ArticleTitle")
        sections = citation.findall("./Article/Abstract/AbstractText")
        if pmid_node is None or pmid_node.text is None or not sections:
            continue

        parts: list[str] = []
        for section in sections:
            body = clean_text("".join(section.itertext()))
            if not body:
                continue
            label = section.get("Label")
            parts.append(f"{label}: {body}" if label else body)

        if not parts:
            continue

        title = clean_text("".join(title_node.itertext())) if title_node is not None else ""
        documents.append(
            Document(
                doc_id=f"pmid:{pmid_node.text.strip()}",
                source="pubmed",
                title=title,
                text=" ".join(parts),
            )
        )

    return documents


def parse_openfda_interactions(payload: str) -> list[str]:
    """Pull the interaction notes out of a label response.

    A label with no `drug_interactions` section yields nothing, rather than a
    note saying there are no interactions. "This label does not list
    interactions" and "this drug has no interactions" are different claims, and
    only the first one is true.
    """
    notes: list[str] = []

    for record in json.loads(payload).get("results", []):
        for section in record.get("drug_interactions", []):
            cleaned = clean_text(section)
            if cleaned:
                notes.append(cleaned)

    return notes


def parse_openfda_labels(payload: str, *, drug: str) -> list[Document]:
    """Turn a label response into documents, one per label record.

    Records carrying none of the sections we keep are dropped for the same
    reason abstract-less PubMed records are: there is no text to ground an
    answer on, and admitting one would let the retriever cite an empty source.
    """
    documents: list[Document] = []

    for ordinal, record in enumerate(json.loads(payload).get("results", [])):
        parts: list[str] = []
        for section in _LABEL_SECTIONS:
            for body in record.get(section, []):
                cleaned = clean_text(body)
                if cleaned:
                    parts.append(f"{section.replace('_', ' ')}: {cleaned}")

        if not parts:
            continue

        # openFDA's `id` is the SPL set id. Falling back to the drug name keeps
        # ids stable across re-ingest, which chunk ids depend on.
        label_id = record.get("id") or f"{drug}-{ordinal}"

        documents.append(
            Document(
                doc_id=f"openfda:{label_id}",
                source="openfda",
                title=f"{drug} label",
                text=" ".join(parts),
            )
        )

    return documents


def fetch_openfda_labels(drug: str, *, limit: int, client: httpx.Client) -> list[Document]:
    """Fetch drug labels for one generic name -- keyless, public domain."""
    response = client.get(
        OPENFDA_LABEL_URL,
        params={"search": f'openfda.generic_name:"{drug}"', "limit": limit},
    )

    # openFDA answers 404 when a search matches nothing. That is not an error
    # here: a drug with no label on file is simply a drug with no label.
    if response.status_code == httpx.codes.NOT_FOUND:
        return []

    response.raise_for_status()

    return parse_openfda_labels(response.text, drug=drug)
