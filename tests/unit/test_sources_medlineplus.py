"""MedlinePlus wsearch parsing.

Two real quirks are reproduced here: the service wraps matched query terms in
`<span class="qt0">` highlight markup, and FullSummary arrives entity-encoded.
Both would otherwise be embedded as if they were clinical prose.
"""

from mesh.retrieval.sources import parse_medlineplus_topics

TOPIC = """<nlmSearchResult><list>
<document rank="0" url="https://medlineplus.gov/highbloodpressure.html">
<content name="title">High &lt;span class="qt0"&gt;Blood Pressure&lt;/span&gt;</content>
<content name="FullSummary">What is blood pressure?&lt;p&gt;It is the force of your
blood.&lt;/p&gt;</content>
</document>
</list></nlmSearchResult>"""

NO_SUMMARY = """<nlmSearchResult><list>
<document rank="0" url="https://medlineplus.gov/empty.html">
<content name="title">A topic with no summary</content>
</document>
</list></nlmSearchResult>"""


def test_a_topic_becomes_one_document_identified_by_its_url():
    docs = parse_medlineplus_topics(TOPIC)

    assert len(docs) == 1
    assert docs[0].doc_id == "https://medlineplus.gov/highbloodpressure.html"
    assert docs[0].source == "medlineplus"


def test_search_highlight_markup_is_stripped_from_the_title():
    docs = parse_medlineplus_topics(TOPIC)

    assert docs[0].title == "High Blood Pressure"


def test_entity_encoded_markup_is_stripped_from_the_summary():
    docs = parse_medlineplus_topics(TOPIC)

    assert docs[0].text == "What is blood pressure? It is the force of your blood."


def test_a_topic_without_a_summary_is_skipped():
    assert parse_medlineplus_topics(NO_SUMMARY) == []


def test_an_empty_result_set_yields_no_documents():
    assert parse_medlineplus_topics("<nlmSearchResult><list></list></nlmSearchResult>") == []
