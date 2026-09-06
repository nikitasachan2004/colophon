"""
Embedder — embeds chunks using sentence-transformers and writes to ChromaDB.

Why all-MiniLM-L6-v2: runs entirely on CPU, no GPU or API key required, small (~80MB),
removes the dependency on Ollama for embedding. Deliberate simplicity choice for
a $0/solo-dev project (architecture.md §2.5).

Why ChromaDB: free, local, persists to disk, no external account or network dependency.
Standard choice for 2026 AI portfolio projects (architecture.md §2.2).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import chromadb

from src.config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL
from src.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)

COLLECTION_NAME = "claude_docs"

# Module-level singletons (lazy-initialized)
_model: SentenceTransformer | None = None
_chroma_client: chromadb.ClientAPI | None = None


class ONNXEmbeddingWrapper:
    """Lightweight ONNX wrapper for all-MiniLM-L6-v2 (~30MB RSS vs ~560MB PyTorch RSS)."""

    def __init__(self):
        import os
        import chromadb.utils.embedding_functions as ef
        import onnxruntime as ort

        self._ef = ef.ONNXMiniLM_L6_V2()
        # Override SessionOptions & prefer INT8 quantized model for ~26MB RSS vs ~168MB FP32 RSS
        quant_path = os.path.join(self._ef.DOWNLOAD_PATH, self._ef.EXTRACTED_FOLDER_NAME, "model_quantized.onnx")
        model_path = quant_path if os.path.exists(quant_path) else os.path.join(self._ef.DOWNLOAD_PATH, self._ef.EXTRACTED_FOLDER_NAME, "model.onnx")
        if not os.path.exists(model_path):
            try:
                import huggingface_hub as hf
                model_path = hf.hf_hub_download(repo_id="xenova/all-MiniLM-L6-v2", filename="onnx/model_quantized.onnx")
            except Exception:
                pass
        if os.path.exists(model_path):
            so = ort.SessionOptions()
            so.log_severity_level = 3
            so.intra_op_num_threads = 1
            so.inter_op_num_threads = 1
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._ef.model = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"], sess_options=so)

    def encode(self, sentences: list[str] | str, show_progress_bar: bool = False):
        if isinstance(sentences, str):
            sentences = [sentences]
        embeddings = self._ef(sentences)
        import numpy as np

        return np.array(embeddings)



def get_embedding_model() -> Any:
    """Lazy-load the embedding model (ONNX runtime preferred for low memory footprint)."""
    global _model
    if _model is None:
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        try:
            _model = ONNXEmbeddingWrapper()
            logger.info("Loaded ONNX embedding model (low-RAM runtime)")
        except Exception as exc:
            logger.warning("Could not load ONNX embedding model (%s), falling back to PyTorch: %s", EMBEDDING_MODEL, exc)
            from sentence_transformers import SentenceTransformer

            _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def get_chroma_client() -> chromadb.ClientAPI:
    """Lazy-load the ChromaDB persistent client."""
    global _chroma_client
    if _chroma_client is None:
        logger.info("Initializing ChromaDB at: %s", CHROMA_PERSIST_DIR)
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return _chroma_client


def get_collection() -> chromadb.Collection:
    """Get or create the main document collection."""
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _chunk_to_id(chunk: Chunk) -> str:
    """
    Generate a deterministic chunk ID from source URL and chunk index.
    Format: claude-<path-slug>-chunk-<index>
    """
    from urllib.parse import urlparse

    path = urlparse(chunk.source_url).path.strip("/").replace("/", "-")
    return f"claude-{path}-chunk-{chunk.chunk_index}"


def embed_chunks(chunks: list[Chunk], batch_size: int = 64) -> list[list[float]]:
    """
    Embed a list of chunks using the sentence-transformers model.
    Returns a list of embedding vectors.
    """
    model = get_embedding_model()
    texts = [c.text for c in chunks]

    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embeddings = model.encode(batch, show_progress_bar=False)
        all_embeddings.extend(embeddings.tolist())
        logger.debug("Embedded batch %d/%d", i // batch_size + 1, (len(texts) - 1) // batch_size + 1)

    return all_embeddings


def write_to_chroma(chunks: list[Chunk], batch_size: int = 100) -> int:
    """
    Embed all chunks and upsert into ChromaDB with full metadata.

    Returns the total number of chunks written.
    """
    if not chunks:
        logger.warning("No chunks to write")
        return 0

    collection = get_collection()
    now = datetime.now(timezone.utc).isoformat()

    logger.info("Embedding %d chunks...", len(chunks))
    embeddings = embed_chunks(chunks, batch_size=batch_size)

    # Prepare data for ChromaDB upsert
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []

    for chunk, embedding in zip(chunks, embeddings):
        chunk_id = _chunk_to_id(chunk)
        ids.append(chunk_id)
        documents.append(chunk.text)
        metadatas.append(
            {
                "source_url": chunk.source_url,
                "section_heading": chunk.section_heading,
                "doc_category": chunk.doc_category,
                "chunk_index": chunk.chunk_index,
                "token_count": chunk.token_count,
                "ingested_at": now,
            }
        )

    # Upsert in batches
    for i in range(0, len(ids), batch_size):
        end = i + batch_size
        collection.upsert(
            ids=ids[i:end],
            documents=documents[i:end],
            embeddings=embeddings[i:end],
            metadatas=metadatas[i:end],
        )
        logger.debug("Upserted batch %d/%d to ChromaDB", i // batch_size + 1, (len(ids) - 1) // batch_size + 1)

    total = collection.count()
    logger.info("ChromaDB now contains %d chunks", total)
    return len(ids)


def run_ingestion(urls: list[str] | None = None) -> int:
    """
    Full ingestion pipeline: scrape → chunk → embed → store.
    Returns the number of chunks ingested.
    """
    from src.ingestion.scraper import scrape_all
    from src.ingestion.chunker import chunk_documents

    logger.info("Starting ingestion pipeline...")
    documents = scrape_all(urls=urls)
    logger.info("Scraped %d documents", len(documents))

    chunks = chunk_documents(documents)
    logger.info("Created %d chunks", len(chunks))

    count = write_to_chroma(chunks)
    logger.info("Ingestion complete: %d chunks stored", count)
    return count


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    count = run_ingestion()
    print(f"Ingestion complete: {count} chunks stored in ChromaDB")
