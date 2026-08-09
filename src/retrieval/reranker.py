"""
Cross-encoder reranker — the precision pass after hybrid retrieval.

Why: bi-encoder/BM25 retrieval is cheap but only "topically similar." A cross-encoder
sees query+chunk together, which is "often the difference between topically similar
and actually answers the question" (architecture.md §2.4). Standard 2026 recipe:
retrieve top-20, rerank to top-5.

Why BGE-reranker-v2: open-source, free, small enough for CPU — keeps us at $0
(architecture.md §2.4). Risk: may be slow on free CPU hosting (tracked in memory.md).
"""

from __future__ import annotations

import logging

from src.config import CONFIDENCE_THRESHOLD, RERANK_CANDIDATE_POOL, RERANK_FINAL_K
from src.retrieval.vector_search import SearchResult

logger = logging.getLogger(__name__)

# Module-level singleton
_reranker = None


def _get_device() -> str:
    """Detect the best device for running the cross-encoder model."""
    import os
    if os.environ.get("FORCE_CPU_RERANKER") == "1":
        return "cpu"
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _get_reranker():
    """Lazy-load the cross-encoder reranker model."""
    global _reranker
    if _reranker is None:
        logger.info("Loading reranker model: BAAI/bge-reranker-v2-m3")
        try:
            from FlagEmbedding import FlagReranker

            device = _get_device()
            logger.info("Initializing FlagReranker on target device: %s", device)
            _reranker = FlagReranker("BAAI/bge-reranker-v2-m3", devices=device, use_fp16=False)
            if device != "cpu":
                _reranker.model.to(device)
            try:
                actual_device = next(_reranker.model.parameters()).device
                logger.info(
                    "FlagReranker loaded successfully. Model parameters confirmed on device: %s",
                    actual_device,
                )
            except Exception as e:
                logger.info("FlagReranker loaded (device check skipped: %s)", e)
        except ImportError:
            logger.warning(
                "FlagEmbedding not installed — falling back to no reranking. "
                "Install with: pip install FlagEmbedding"
            )
            _reranker = "unavailable"
        except Exception as exc:
            logger.warning("Failed to load reranker: %s — falling back", exc)
            _reranker = "unavailable"
    return _reranker


def rerank(
    query: str,
    candidates: list[SearchResult],
    final_k: int | None = None,
    confidence_threshold: float | None = None,
    candidate_pool: int | None = None,
) -> tuple[list[SearchResult], bool]:
    """
    Rerank candidates using the cross-encoder, return top-k.

    Also implements the "no context found" check (FR-10, design.md §4):
    if the top reranked chunk's score is below the confidence threshold,
    context_found is set to False.

    Args:
        query: The user's natural-language question.
        candidates: Pre-fused candidate SearchResults.
        final_k: Number of results to keep after reranking.
        confidence_threshold: Score below which we declare "no context found."
        candidate_pool: Max number of candidates passed into cross-encoder (Option B).

    Returns:
        (top_k_results, context_found) — context_found is False if the best
        result is below the confidence threshold.
    """
    final_k = final_k or RERANK_FINAL_K
    confidence_threshold = confidence_threshold or CONFIDENCE_THRESHOLD
    candidate_pool = candidate_pool or RERANK_CANDIDATE_POOL

    if not candidates:
        return [], False

    # Option B: Cap candidates to top candidate_pool to prevent latency blowup
    if len(candidates) > candidate_pool:
        logger.debug(
            "Capping reranker candidates from %d to top %d", len(candidates), candidate_pool
        )
        candidates = candidates[:candidate_pool]

    reranker = _get_reranker()

    # Fallback if reranker unavailable: just return top-k by existing scores
    if reranker == "unavailable":
        logger.warning("Reranker unavailable, using pre-fusion scores")
        top = sorted(candidates, key=lambda r: r.score, reverse=True)[:final_k]
        context_found = len(top) > 0 and top[0].score > 0
        return top, context_found

    # Build query-passage pairs for the cross-encoder
    pairs = [[query, c.text] for c in candidates]

    try:
        scores = reranker.compute_score(pairs, normalize=True)
        # compute_score returns a single float if only one pair
        if isinstance(scores, (int, float)):
            scores = [scores]
    except Exception as exc:
        logger.error("Reranker scoring failed: %s — falling back", exc)
        top = sorted(candidates, key=lambda r: r.score, reverse=True)[:final_k]
        context_found = len(top) > 0 and top[0].score > 0
        return top, context_found

    # Attach reranker scores to candidates
    scored = list(zip(candidates, scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    top_results: list[SearchResult] = []
    for candidate, score in scored[:final_k]:
        top_results.append(
            SearchResult(
                chunk_id=candidate.chunk_id,
                text=candidate.text,
                score=float(score),
                source_url=candidate.source_url,
                section_heading=candidate.section_heading,
                doc_category=candidate.doc_category,
            )
        )

    # "No context found" check — FR-10
    context_found = len(top_results) > 0 and top_results[0].score >= confidence_threshold

    logger.debug(
        "Reranked %d → %d candidates (top score: %.3f, threshold: %.3f, context_found: %s)",
        len(candidates),
        len(top_results),
        top_results[0].score if top_results else 0,
        confidence_threshold,
        context_found,
    )
    return top_results, context_found
