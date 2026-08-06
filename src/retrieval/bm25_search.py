"""
BM25 sparse keyword search.

Why hybrid (not vector-only): vector search is good at meaning but misses exact-match
terms — critical for API docs where a user might type exact method names like
`client.messages.create` or `tool_choice`. BM25 catches these reliably where dense embeddings
sometimes blur (architecture.md §2.3).
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.config import PROCESSED_DATA_DIR
from src.ingestion.embedder import get_collection
from src.retrieval.vector_search import SearchResult

logger = logging.getLogger(__name__)

BM25_INDEX_PATH = PROCESSED_DATA_DIR / "bm25_index.pkl"

# Module-level cache
_bm25_index: BM25Okapi | None = None
_bm25_corpus_ids: list[str] = []
_bm25_corpus_texts: list[str] = []
_bm25_corpus_metadata: list[dict] = []


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer for BM25."""
    import re

    text = text.lower()
    # Keep alphanumeric, dots (for API names), underscores, hyphens
    tokens = re.findall(r"[a-z0-9_.]+(?:\.[a-z0-9_]+)*", text)
    return tokens


def build_bm25_index() -> BM25Okapi:
    """
    Build a BM25 index from all chunks in ChromaDB.
    Caches to disk for fast reloads.
    """
    global _bm25_index, _bm25_corpus_ids, _bm25_corpus_texts, _bm25_corpus_metadata

    collection = get_collection()
    total = collection.count()
    if total == 0:
        raise ValueError("ChromaDB is empty — run ingestion first")

    # Fetch all documents from ChromaDB
    all_data = collection.get(include=["documents", "metadatas"])

    _bm25_corpus_ids = all_data["ids"]
    _bm25_corpus_texts = all_data["documents"]
    _bm25_corpus_metadata = all_data["metadatas"]

    # Tokenize corpus for BM25
    tokenized_corpus = [_tokenize(text) for text in _bm25_corpus_texts]
    _bm25_index = BM25Okapi(tokenized_corpus)

    # Cache to disk
    cache_data = {
        "ids": _bm25_corpus_ids,
        "texts": _bm25_corpus_texts,
        "metadata": _bm25_corpus_metadata,
    }
    BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(cache_data, f)

    logger.info("Built BM25 index over %d documents", total)
    return _bm25_index


def _ensure_index_loaded() -> None:
    """Load BM25 index from cache or rebuild from ChromaDB."""
    global _bm25_index, _bm25_corpus_ids, _bm25_corpus_texts, _bm25_corpus_metadata

    if _bm25_index is not None:
        return

    if BM25_INDEX_PATH.exists():
        logger.info("Loading cached BM25 index from disk...")
        with open(BM25_INDEX_PATH, "rb") as f:
            cache_data = pickle.load(f)
        _bm25_corpus_ids = cache_data["ids"]
        _bm25_corpus_texts = cache_data["texts"]
        _bm25_corpus_metadata = cache_data["metadata"]
        tokenized = [_tokenize(text) for text in _bm25_corpus_texts]
        _bm25_index = BM25Okapi(tokenized)
        logger.info("BM25 index loaded: %d documents", len(_bm25_corpus_ids))
    else:
        build_bm25_index()


def bm25_search(query: str, top_k: int = 20) -> list[SearchResult]:
    """
    Sparse keyword search using BM25. Returns top-k results ranked by
    keyword relevance score.
    """
    _ensure_index_loaded()
    assert _bm25_index is not None

    tokenized_query = _tokenize(query)
    scores = _bm25_index.get_scores(tokenized_query)

    # Get top-k indices sorted by score (descending)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
        :top_k
    ]

    results: list[SearchResult] = []
    for idx in top_indices:
        if scores[idx] <= 0:
            continue  # Skip zero-score results

        metadata = _bm25_corpus_metadata[idx] if idx < len(_bm25_corpus_metadata) else {}
        results.append(
            SearchResult(
                chunk_id=_bm25_corpus_ids[idx],
                text=_bm25_corpus_texts[idx],
                score=float(scores[idx]),
                source_url=metadata.get("source_url", ""),
                section_heading=metadata.get("section_heading", ""),
                doc_category=metadata.get("doc_category", ""),
            )
        )

    logger.debug("BM25 search returned %d results", len(results))
    return results
