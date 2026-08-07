"""
Reciprocal Rank Fusion — merges dense and sparse search results.

Why RRF over simple score averaging: dense and sparse scores are on completely
different scales (cosine similarity vs BM25 term frequency). RRF works with
ranks, not raw scores, so it naturally handles this mismatch. Standard 2026
production recipe (architecture.md §2.3, design.md §4).
"""

from __future__ import annotations

import logging
from collections import defaultdict

from src.retrieval.vector_search import SearchResult

logger = logging.getLogger(__name__)

# RRF constant — standard value from the original RRF paper (Cormack et al. 2009)
RRF_K_CONSTANT = 60


def reciprocal_rank_fusion(
    dense_results: list[SearchResult],
    sparse_results: list[SearchResult],
    k_constant: int = RRF_K_CONSTANT,
) -> list[SearchResult]:
    """
    Fuse two ranked result lists using Reciprocal Rank Fusion.

    RRF score for each document = sum over each list of 1 / (k + rank).
    This gives a single ranked list that benefits from both search strategies.

    Args:
        dense_results: Results from vector search, sorted by score desc.
        sparse_results: Results from BM25 search, sorted by score desc.
        k_constant: Smoothing constant (default 60, standard value).

    Returns:
        Fused and deduplicated list of SearchResult, sorted by RRF score desc.
    """
    rrf_scores: dict[str, float] = defaultdict(float)
    result_lookup: dict[str, SearchResult] = {}

    # Score from dense (vector) results
    for rank, result in enumerate(dense_results):
        rrf_scores[result.chunk_id] += 1.0 / (k_constant + rank)
        result_lookup[result.chunk_id] = result

    # Score from sparse (BM25) results
    for rank, result in enumerate(sparse_results):
        rrf_scores[result.chunk_id] += 1.0 / (k_constant + rank)
        # Prefer dense result metadata if both exist (same chunk)
        if result.chunk_id not in result_lookup:
            result_lookup[result.chunk_id] = result

    # Sort by fused score, descending
    sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)

    fused: list[SearchResult] = []
    for chunk_id in sorted_ids:
        original = result_lookup[chunk_id]
        fused.append(
            SearchResult(
                chunk_id=chunk_id,
                text=original.text,
                score=rrf_scores[chunk_id],  # RRF score, not original score
                source_url=original.source_url,
                section_heading=original.section_heading,
                doc_category=original.doc_category,
            )
        )

    logger.debug(
        "RRF fused %d dense + %d sparse → %d unique results",
        len(dense_results),
        len(sparse_results),
        len(fused),
    )
    return fused
