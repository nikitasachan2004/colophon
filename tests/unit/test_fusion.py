"""
Unit tests for Reciprocal Rank Fusion (RRF).

Validates:
1. Merging of disjoint dense and sparse candidate lists
2. Proper rank scoring and ordering
3. Deduplication of chunks appearing in both lists
"""

import pytest
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.vector_search import SearchResult


def test_rrf_scoring_and_deduplication():
    # Item A is rank 0 in dense, rank 0 in sparse
    # Item B is rank 1 in dense only
    # Item C is rank 1 in sparse only
    dense = [
        SearchResult("chunk-A", "text A", 0.95, "url-A", "Heading A", "cat1"),
        SearchResult("chunk-B", "text B", 0.85, "url-B", "Heading B", "cat1"),
    ]
    sparse = [
        SearchResult("chunk-A", "text A", 12.5, "url-A", "Heading A", "cat1"),
        SearchResult("chunk-C", "text C", 8.0, "url-C", "Heading C", "cat2"),
    ]

    fused = reciprocal_rank_fusion(dense, sparse, k_constant=60)

    assert len(fused) == 3
    # chunk-A must be ranked #1 because it appeared at rank 0 in both lists
    assert fused[0].chunk_id == "chunk-A"
    expected_score_a = (1.0 / 60.0) + (1.0 / 60.0)
    assert abs(fused[0].score - expected_score_a) < 1e-6

    # chunk-B and chunk-C each have 1/61
    assert {fused[1].chunk_id, fused[2].chunk_id} == {"chunk-B", "chunk-C"}


def test_rrf_empty_inputs():
    assert reciprocal_rank_fusion([], []) == []

    dense_only = [
        SearchResult("chunk-1", "text 1", 0.9, "url-1", "Heading 1", "cat1")
    ]
    fused = reciprocal_rank_fusion(dense_only, [])
    assert len(fused) == 1
    assert fused[0].chunk_id == "chunk-1"
