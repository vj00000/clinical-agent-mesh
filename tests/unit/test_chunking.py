"""Chunking behaviour.

Chunk ids must be stable across re-ingestion: a citation stored in an eval
result or a conversation checkpoint has to still resolve after the corpus is
rebuilt.
"""

from mesh.retrieval.chunking import chunk_document

WORDS = " ".join(f"w{i}" for i in range(250))


def test_long_text_is_split_into_chunks_of_at_most_the_target_size():
    chunks = chunk_document(WORDS, source="cdc-htn", target_words=100, overlap_words=20)

    assert len(chunks) > 1
    assert all(len(c.text.split()) <= 100 for c in chunks)


def test_consecutive_chunks_overlap_by_the_configured_window():
    chunks = chunk_document(WORDS, source="cdc-htn", target_words=100, overlap_words=20)

    first_tail = chunks[0].text.split()[-20:]
    second_head = chunks[1].text.split()[:20]
    assert first_tail == second_head


def test_text_shorter_than_the_target_yields_a_single_chunk():
    chunks = chunk_document("short guideline text", source="cdc-htn", target_words=100)

    assert len(chunks) == 1
    assert chunks[0].text == "short guideline text"


def test_chunk_ids_are_stable_across_runs_for_identical_input():
    first = chunk_document(WORDS, source="cdc-htn", target_words=100, overlap_words=20)
    second = chunk_document(WORDS, source="cdc-htn", target_words=100, overlap_words=20)

    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]


def test_same_text_from_different_sources_gets_different_chunk_ids():
    cdc = chunk_document(WORDS, source="cdc-htn", target_words=100, overlap_words=20)
    who = chunk_document(WORDS, source="who-htn", target_words=100, overlap_words=20)

    assert cdc[0].chunk_id != who[0].chunk_id


def test_chunks_record_their_position_in_the_document():
    chunks = chunk_document(WORDS, source="cdc-htn", target_words=100, overlap_words=20)

    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
