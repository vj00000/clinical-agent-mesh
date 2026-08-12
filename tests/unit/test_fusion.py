"""Reciprocal rank fusion.

RRF merges the dense and keyword result lists without needing their scores to
be on a comparable scale, which is why it is preferred over raw score addition.
"""

from mesh.retrieval.fusion import reciprocal_rank_fusion


def test_a_document_found_by_both_retrievers_outranks_one_found_by_only_one():
    dense = ["a", "b", "c"]
    keyword = ["c", "d", "e"]

    fused = reciprocal_rank_fusion([dense, keyword])

    assert fused[0] == "c"


def test_within_a_single_list_the_original_order_is_preserved():
    fused = reciprocal_rank_fusion([["a", "b", "c"]])

    assert fused == ["a", "b", "c"]


def test_ties_are_broken_by_first_appearance_so_results_are_deterministic():
    fused = reciprocal_rank_fusion([["a"], ["b"]])

    assert fused == ["a", "b"]


def test_no_results_fuse_to_nothing():
    assert reciprocal_rank_fusion([[], []]) == []


def test_the_fused_list_contains_no_duplicates():
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "a"]])

    assert sorted(fused) == ["a", "b"]
