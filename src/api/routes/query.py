"""
Query route — the core POST /query endpoint.

Implements the full query-time pipeline: embed → hybrid retrieve → fuse → rerank
→ generate → return with sources and latency (architecture.md §3, design.md §9).
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter

from src.api.schemas import QueryRequest, QueryResponse, SourceChunk
from src.config import RERANK_FINAL_K, RETRIEVAL_TOP_K
from src.generation import llm_client
from src.generation.prompts import NO_CONTEXT_RESPONSE, SYSTEM_PROMPT, build_user_prompt
from src.retrieval.bm25_search import bm25_search
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.reranker import rerank
from src.retrieval.vector_search import vector_search

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """
    Full RAG pipeline — data flow per architecture.md §3:

    1. Query embedded + dense search (top-20)
    2. BM25 sparse search (top-20)
    3. RRF fusion → single ranked list
    4. Cross-encoder reranking → top-5
    5. Prompt assembly with citation metadata
    6. LLM generation (Ollama or Groq)
    7. Return answer + sources + latency
    """
    t_start = time.perf_counter()

    top_k = request.top_k or RERANK_FINAL_K
    retrieval_k = RETRIEVAL_TOP_K

    # Step 1 & 2: Hybrid retrieval
    t0 = time.perf_counter()
    dense_results = vector_search(request.question, top_k=retrieval_k)
    sparse_results = bm25_search(request.question, top_k=retrieval_k)
    retrieval_ms = int((time.perf_counter() - t0) * 1000)

    # Step 3: Reciprocal Rank Fusion
    t0 = time.perf_counter()
    fused = reciprocal_rank_fusion(dense_results, sparse_results)
    fusion_ms = int((time.perf_counter() - t0) * 1000)

    # Step 4: Reranking + confidence threshold check (FR-10)
    t0 = time.perf_counter()
    top_results, context_found = rerank(
        query=request.question,
        candidates=fused,
        final_k=top_k,
    )
    rerank_ms = int((time.perf_counter() - t0) * 1000)

    # Step 5 & 6: Generate answer
    backend = llm_client.get_active_backend()
    t0 = time.perf_counter()

    if not context_found or not top_results:
        # No relevant context — use the explicit refusal (FR-10)
        answer = NO_CONTEXT_RESPONSE
        sources: list[SourceChunk] = []
        gen_ms = int((time.perf_counter() - t0) * 1000)
    else:
        # Build prompt with top chunks
        chunk_dicts = [
            {
                "text": r.text,
                "section_heading": r.section_heading,
                "source_url": r.source_url,
            }
            for r in top_results
        ]
        user_prompt = build_user_prompt(request.question, chunk_dicts)

        try:
            answer = llm_client.generate(SYSTEM_PROMPT, user_prompt)
        except Exception as exc:
            logger.error("LLM generation failed: %s", exc)
            answer = f"Error generating answer: {exc}"
        gen_ms = int((time.perf_counter() - t0) * 1000)

        sources = [
            SourceChunk(
                text=r.text[:500],  # Truncate for response size
                source_url=r.source_url,
                section_heading=r.section_heading,
                relevance_score=round(r.score, 4),
            )
            for r in top_results
        ]

    elapsed_ms = int((time.perf_counter() - t_start) * 1000)

    logger.info(
        "Query processed in %dms [retrieval=%dms, fusion=%dms, rerank=%dms, generation=%dms] (backend=%s, context_found=%s, sources=%d)",
        elapsed_ms,
        retrieval_ms,
        fusion_ms,
        rerank_ms,
        gen_ms,
        backend,
        context_found,
        len(sources),
    )

    return QueryResponse(
        answer=answer,
        sources=sources,
        latency_ms=elapsed_ms,
        backend_used=backend,
        context_found=context_found,
    )
