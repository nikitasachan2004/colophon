"""
Build the 'baseline_naive_chunks' ChromaDB collection for Phase 4 ablation Stage 1.

Why this exists (design.md §6, memory.md Phase 4 redo):
  The last ablation run's "Baseline" and "+Structure-Aware Chunking" rows both
  queried the SAME production collection — making them score identically (0.800
  Recall@5 each) and the comparison meaningless.  A genuine ablation requires a
  genuinely different baseline index.

This script creates that baseline: fixed-size chunking (512 tokens, no heading
awareness, no code-block protection, 10% overlap) — exactly what you'd write on
day-1 before thinking about document structure.  It writes to a SEPARATE collection
('baseline_naive_chunks') so the production 'claude_docs' collection is never touched.

Run once before the ablation study:
  python -m src.evaluation.build_baseline_collection

Safe to re-run: uses upsert, so running it twice produces the same result.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb
import tiktoken
from sentence_transformers import SentenceTransformer

from src.config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL, RAW_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger("build_baseline")

# ── Constants ─────────────────────────────────────────────────────────────────

BASELINE_COLLECTION_NAME = "baseline_naive_chunks"
CHUNK_SIZE_TOKENS = 512      # same as production — only chunking STRATEGY differs
CHUNK_OVERLAP_TOKENS = 51    # 10% overlap (512 * 0.10 ≈ 51)

_TOKENIZER = tiktoken.get_encoding("cl100k_base")


# ── Naive fixed-size chunker ──────────────────────────────────────────────────

def _tokenize(text: str) -> list[int]:
    return _TOKENIZER.encode(text)


def _decode(token_ids: list[int]) -> str:
    return _TOKENIZER.decode(token_ids)


def naive_fixed_size_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE_TOKENS,
    overlap: int = CHUNK_OVERLAP_TOKENS,
) -> list[str]:
    """
    Naive fixed-size chunking: split on token count, slide with overlap.
    No heading awareness. No code-block protection. Just tokens.

    This is the baseline: what you'd write before thinking about document structure.
    """
    tokens = _tokenize(text)
    if not tokens:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = _decode(chunk_tokens).strip()
        if chunk_text:
            chunks.append(chunk_text)
        if end >= len(tokens):
            break
        start = end - overlap  # slide back by overlap for next window

    return chunks


# ── Front-matter parser ───────────────────────────────────────────────────────

def _infer_url_from_filename(path: Path) -> str | None:
    """
    Infer source URL from raw filename for API reference files that have no front-matter.

    Filename pattern: en_<path_segments...>_<16-char-hex-hash>.md
    e.g. en_api_files_1d4befac3caeef9c.md -> /docs/en/api/files
         en_api_messages_count_tokens_c03.md -> /docs/en/api/messages/count_tokens

    Special case: count_tokens must stay as one segment (not count/tokens).
    """
    BASE = "https://platform.claude.com/docs/"
    name = path.stem  # without .md
    # Strip trailing 16-char hex hash
    without_hash = re.sub(r"_[0-9a-f]{16}$", "", name)
    parts = without_hash.split("_")

    # Reconstruct: look for known multi-word segments that must not be split
    # count_tokens: segments [..., 'count', 'tokens'] -> keep as 'count_tokens'
    merged: list[str] = []
    i = 0
    while i < len(parts):
        if i + 1 < len(parts) and parts[i] == "count" and parts[i + 1] == "tokens":
            merged.append("count_tokens")
            i += 2
        elif i + 1 < len(parts) and parts[i] == "retrieve" and parts[i + 1] == "metadata":
            merged.append("retrieve_metadata")
            i += 2
        else:
            merged.append(parts[i])
            i += 1

    return BASE + "/".join(merged)


def _parse_raw_file(path: Path) -> dict[str, str] | None:
    """
    Parse a raw scraped markdown file.

    Handles two formats:
    1. Front-matter block (--- title/url/description --- then body): ~58 files
    2. No front-matter (API reference pages saved as raw markdown): ~19 files
       URL is inferred from the filename.

    Returns dict with keys: url, category, text — or None if unparseable.
    """
    content = path.read_text(encoding="utf-8")

    # Try front-matter first
    fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
    if fm_match:
        fm = fm_match.group(1)
        url_match = re.search(r"^url:\s*(.+)$", fm, re.MULTILINE)
        if not url_match:
            logger.warning("Front-matter has no url field in %s — skipping", path.name)
            return None
        url = url_match.group(1).strip()
        body = content[fm_match.end():]
    else:
        # No front-matter: infer URL from filename (API reference files)
        url = _infer_url_from_filename(path)
        if url is None:
            logger.warning("Cannot infer URL for %s — skipping", path.name)
            return None
        body = content
        logger.debug("Inferred URL for %s -> %s", path.name, url)

    category = _url_to_category(url)
    return {"url": url, "category": category, "text": body}


def _url_to_category(url: str) -> str:
    """Derive a coarse doc_category from the URL — mirrors the production scraper."""
    path = url.split("platform.claude.com/docs/en/")[-1] if "platform.claude.com" in url else url
    if path.startswith("agents-and-tools/tool-use"):
        return "tool_use"
    if path.startswith("agents-and-tools"):
        return "agents_and_tools"
    if path.startswith("build-with-claude/extended-thinking") or "thinking" in path:
        return "extended_thinking"
    if path.startswith("build-with-claude"):
        return "messages_api"
    if path.startswith("api/"):
        return "api_reference"
    if path.startswith("about-claude/models"):
        return "models"
    if path.startswith("about-claude"):
        return "about_claude"
    if path.startswith("test-and-evaluate"):
        return "testing"
    if path.startswith("manage-claude"):
        return "management"
    return "general"


# ── Chunk dataclass (mirrors src.ingestion.chunker.Chunk) ────────────────────

class NaiveChunk:
    __slots__ = ("text", "source_url", "doc_category", "chunk_index", "token_count")

    def __init__(
        self,
        text: str,
        source_url: str,
        doc_category: str,
        chunk_index: int,
    ) -> None:
        self.text = text
        self.source_url = source_url
        self.doc_category = doc_category
        self.chunk_index = chunk_index
        self.token_count = len(_tokenize(text))


def _chunk_to_id(chunk: NaiveChunk) -> str:
    from urllib.parse import urlparse
    path = urlparse(chunk.source_url).path.strip("/").replace("/", "-")
    return f"baseline-{path}-chunk-{chunk.chunk_index}"


# ── Ingestion ─────────────────────────────────────────────────────────────────

def build_baseline_collection() -> int:
    """
    Read all raw markdown files, naive-chunk them, embed, and write to
    'baseline_naive_chunks' ChromaDB collection.

    Returns the total chunk count written.
    Does NOT touch 'claude_docs' (the production collection).
    """
    raw_files = sorted(RAW_DATA_DIR.glob("*.md"))
    if not raw_files:
        raise FileNotFoundError(
            f"No .md files found in {RAW_DATA_DIR}. "
            "Run the scraper first: python -m src.ingestion.embedder"
        )

    logger.info("Found %d raw markdown files in %s", len(raw_files), RAW_DATA_DIR)

    # ── Parse raw files ───────────────────────────────────────────────────────
    all_chunks: list[NaiveChunk] = []
    docs_processed = 0
    docs_skipped = 0

    for path in raw_files:
        doc = _parse_raw_file(path)
        if doc is None:
            docs_skipped += 1
            continue

        chunk_texts = naive_fixed_size_chunks(doc["text"])
        for idx, text in enumerate(chunk_texts):
            all_chunks.append(
                NaiveChunk(
                    text=text,
                    source_url=doc["url"],
                    doc_category=doc["category"],
                    chunk_index=idx,
                )
            )
        docs_processed += 1

    logger.info(
        "Parsed %d docs (%d skipped) → %d naive fixed-size chunks",
        docs_processed,
        docs_skipped,
        len(all_chunks),
    )

    if not all_chunks:
        raise ValueError("No chunks produced — check raw data and chunker")

    # ── Embed ─────────────────────────────────────────────────────────────────
    logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [c.text for c in all_chunks]
    logger.info("Embedding %d chunks (batch_size=64)...", len(all_chunks))
    embeddings_np = model.encode(texts, batch_size=64, show_progress_bar=True)
    embeddings: list[list[float]] = embeddings_np.tolist()
    logger.info("Embedding complete")

    # ── Write to ChromaDB ─────────────────────────────────────────────────────
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    # Delete and recreate for a clean run (ablation requires a fresh baseline)
    existing = [c.name for c in client.list_collections()]
    if BASELINE_COLLECTION_NAME in existing:
        logger.info(
            "Collection '%s' already exists (%d chunks) — deleting for fresh build",
            BASELINE_COLLECTION_NAME,
            client.get_collection(BASELINE_COLLECTION_NAME).count(),
        )
        client.delete_collection(BASELINE_COLLECTION_NAME)

    collection = client.create_collection(
        name=BASELINE_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info("Created collection '%s'", BASELINE_COLLECTION_NAME)

    BATCH = 100
    now = datetime.now(timezone.utc).isoformat()
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []

    for chunk in all_chunks:
        ids.append(_chunk_to_id(chunk))
        documents.append(chunk.text)
        metadatas.append({
            "source_url":      chunk.source_url,
            "section_heading": "",           # no heading awareness — that's the point
            "doc_category":    chunk.doc_category,
            "chunk_index":     chunk.chunk_index,
            "token_count":     chunk.token_count,
            "ingested_at":     now,
            "chunking_strategy": "naive_fixed_size",
        })

    for i in range(0, len(ids), BATCH):
        collection.upsert(
            ids=ids[i : i + BATCH],
            documents=documents[i : i + BATCH],
            embeddings=embeddings[i : i + BATCH],
            metadatas=metadatas[i : i + BATCH],
        )
        logger.debug("Upserted batch %d/%d", i // BATCH + 1, (len(ids) - 1) // BATCH + 1)

    final_count = collection.count()
    logger.info(
        "Baseline collection '%s' built: %d chunks (production 'claude_docs': untouched)",
        BASELINE_COLLECTION_NAME,
        final_count,
    )
    return final_count


if __name__ == "__main__":
    count = build_baseline_collection()
    print(f"\nBaseline collection built: {count} naive fixed-size chunks")
    print(f"Collection name: '{BASELINE_COLLECTION_NAME}'")
    print("Production 'claude_docs' collection: untouched")
