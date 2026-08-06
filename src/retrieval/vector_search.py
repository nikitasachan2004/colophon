"""
Dense vector search against ChromaDB.

Why: cosine similarity search on embedded chunks is one half of the hybrid
retrieval strategy (architecture.md §2.3). Returns top-k candidates ranked
by semantic similarity to the query.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.ingestion.embedder import get_collection, get_embedding_model

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result with score and metadata."""

    chunk_id: str
    text: str
    score: float
    source_url: str
    section_heading: str
    doc_category: str


def vector_search(query: str, top_k: int = 20) -> list[SearchResult]:
    """
    Dense vector search: embed the query with the same model used for ingestion,
    then find the top-k most similar chunks in ChromaDB (cosine similarity).
    """
    collection = get_collection()
    count = collection.count()
    if count == 0:
        logger.warning("Vector search called on empty collection")
        return []

    model = get_embedding_model()
    query_embedding = model.encode([query], show_progress_bar=False).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )

    if not results["ids"] or not results["ids"][0]:
        logger.warning("Vector search returned no results for: %s", query[:80])
        return []

    search_results: list[SearchResult] = []
    for i, chunk_id in enumerate(results["ids"][0]):
        # ChromaDB returns distances (lower = more similar for cosine),
        # convert to a similarity score
        distance = results["distances"][0][i] if results["distances"] else 0.0
        score = 1.0 - distance  # cosine similarity = 1 - cosine distance

        metadata = results["metadatas"][0][i] if results["metadatas"] else {}
        text = results["documents"][0][i] if results["documents"] else ""

        search_results.append(
            SearchResult(
                chunk_id=chunk_id,
                text=text,
                score=score,
                source_url=metadata.get("source_url", ""),
                section_heading=metadata.get("section_heading", ""),
                doc_category=metadata.get("doc_category", ""),
            )
        )

    logger.debug("Vector search returned %d results", len(search_results))
    return search_results
