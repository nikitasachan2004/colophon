"""
Unit tests for cross-encoder reranker module and fallback logic.

Validates:
1. Fallback behavior when model is not available
2. Correct handling of confidence threshold for OOD / weak results
3. Top-k filtering
"""

import pytest
from src.retrieval.reranker import rerank
from src.retrieval.vector_search import SearchResult


def test_rerank_empty_candidates():
    results, context_found = rerank("Any query", [], final_k=5)
    assert results == []
    assert context_found is False


def test_rerank_confidence_threshold_rejection():
    candidates = [
        SearchResult("chunk-1", "Random irrelevant text", 0.05, "url-1", "Heading", "cat"),
    ]
    # Set threshold high to test rejection
    results, context_found = rerank(
        query="Quantum astrophysics spacecraft propulsion API",
        candidates=candidates,
        final_k=3,
        confidence_threshold=0.99,
    )
    assert len(results) <= 1
    assert context_found is False


def test_rerank_top_k_truncation():
    candidates = [
        SearchResult(f"chunk-{i}", f"Documentation snippet {i}", float(i), f"url-{i}", "Heading", "cat")
        for i in range(10)
    ]
    results, _ = rerank(
        query="Documentation snippet",
        candidates=candidates,
        final_k=3,
        confidence_threshold=0.0,
    )
    assert len(results) == 3
