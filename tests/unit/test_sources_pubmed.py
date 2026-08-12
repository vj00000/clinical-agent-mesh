"""PubMed efetch parsing.

The fixtures mirror the real payload shape: PMIDs carry a Version attribute and
abstracts arrive as several Label-tagged AbstractText sections rather than one
block of prose.
"""

from mesh.retrieval.sources import parse_esearch_ids, parse_pubmed_articles

ESEARCH = """{"header":{"type":"esearch"},"esearchresult":{"count":"12211","retmax":"3",
"idlist":["42581634","42580367","42576812"]}}"""

TWO_SECTION_ARTICLE = """<PubmedArticleSet><PubmedArticle><MedlineCitation>
<PMID Version="1">42581634</PMID>
<Article>
<ArticleTitle>Quality of Diabetes Care Among Older Adults</ArticleTitle>
<Abstract>
<AbstractText Label="AIMS" NlmCategory="OBJECTIVE">To evaluate care quality.</AbstractText>
<AbstractText Label="RESULTS" NlmCategory="RESULTS">Care improved modestly.</AbstractText>
</Abstract>
</Article>
</MedlineCitation></PubmedArticle></PubmedArticleSet>"""

UNLABELLED_ARTICLE = """<PubmedArticleSet><PubmedArticle><MedlineCitation>
<PMID Version="1">111</PMID>
<Article>
<ArticleTitle>A short report</ArticleTitle>
<Abstract><AbstractText>Single block of abstract text.</AbstractText></Abstract>
</Article>
</MedlineCitation></PubmedArticle></PubmedArticleSet>"""

NO_ABSTRACT = """<PubmedArticleSet><PubmedArticle><MedlineCitation>
<PMID Version="1">222</PMID>
<Article><ArticleTitle>Editorial with no abstract</ArticleTitle></Article>
</MedlineCitation></PubmedArticle></PubmedArticleSet>"""


def test_an_article_becomes_one_document_identified_by_its_pmid():
    docs = parse_pubmed_articles(TWO_SECTION_ARTICLE)

    assert len(docs) == 1
    assert docs[0].doc_id == "pmid:42581634"
    assert docs[0].source == "pubmed"


def test_the_article_title_is_captured():
    docs = parse_pubmed_articles(TWO_SECTION_ARTICLE)

    assert docs[0].title == "Quality of Diabetes Care Among Older Adults"


def test_labelled_abstract_sections_keep_their_labels_in_the_text():
    """Dropping the labels would lose the distinction between aims and findings."""
    docs = parse_pubmed_articles(TWO_SECTION_ARTICLE)

    assert docs[0].text == "AIMS: To evaluate care quality. RESULTS: Care improved modestly."


def test_an_unlabelled_abstract_is_kept_as_plain_text():
    docs = parse_pubmed_articles(UNLABELLED_ARTICLE)

    assert docs[0].text == "Single block of abstract text."


def test_an_article_without_an_abstract_is_skipped():
    """There is nothing to ground an answer on, so it must not enter the corpus."""
    assert parse_pubmed_articles(NO_ABSTRACT) == []


def test_an_empty_result_set_yields_no_documents():
    assert parse_pubmed_articles("<PubmedArticleSet></PubmedArticleSet>") == []


def test_esearch_ids_are_extracted_in_order():
    assert parse_esearch_ids(ESEARCH) == ["42581634", "42580367", "42576812"]


def test_an_esearch_response_with_no_hits_yields_no_ids():
    assert parse_esearch_ids('{"esearchresult":{"count":"0","idlist":[]}}') == []
