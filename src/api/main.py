"""
FastAPI application — the backend API for the RAG Knowledge Assistant.

Why FastAPI: async support for I/O-bound LLM/embedding calls, automatic interactive
API docs at /docs, and the dominant choice across 2026 AI-engineer project references
(architecture.md §2.7).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import health, query
from src.config import ENABLE_ADMIN

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: pre-load models and indexes so the first query isn't slow.
    Shutdown: clean up resources.
    """
    logger.info("Starting RAG Knowledge Assistant API...")

    # Limit PyTorch threads to save memory on 512MB instances
    try:
        import torch
        torch.set_num_threads(1)
        import os
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
    except Exception:
        pass

    # Pre-load embedding model
    try:
        from src.ingestion.embedder import get_embedding_model, get_collection

        get_embedding_model()
        collection = get_collection()
        logger.info("ChromaDB loaded: %d chunks indexed", collection.count())
    except Exception as exc:
        logger.warning("Could not pre-load embeddings/ChromaDB: %s", exc)

    # Pre-load BM25 index
    try:
        from src.retrieval.bm25_search import _ensure_index_loaded

        _ensure_index_loaded()
    except Exception as exc:
        logger.warning("Could not pre-load BM25 index: %s", exc)

    # Pre-load cross-encoder reranker model
    try:
        from src.retrieval.reranker import _get_reranker

        _get_reranker()
        logger.info("BGE reranker model pre-loaded successfully")
    except Exception as exc:
        logger.warning("Could not pre-load reranker: %s", exc)

    yield

    logger.info("Shutting down RAG Knowledge Assistant API")


app = FastAPI(
    title="Colophon — Claude API Knowledge Assistant",
    description="Retrieval-Augmented Generation system for Anthropic Claude API documentation. Grounds every answer in cited source chunks with zero hallucination.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — explicit allowlist.
# "*.vercel.app" covers preview deployments; the production domain should be
# added to ALLOWED_ORIGINS in the environment once the Vercel project is named.
# localhost:3000 covers Next.js dev server; localhost:8501 keeps Streamlit working locally.
import os as _os

_EXTRA = [o.strip() for o in _os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]

ALLOWED_ORIGINS = [
    "http://localhost:3000",        # Next.js dev
    "http://localhost:8501",        # Streamlit dev (legacy)
    "https://*.vercel.app",         # Vercel preview & prod deployments
    *_EXTRA,                        # Any domain injected via ALLOWED_ORIGINS env var
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"https://.*\.vercel\.app",  # Covers all Vercel preview URLs
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# Register routes
app.include_router(query.router, tags=["Query"])
app.include_router(health.router, tags=["Health"])


# Admin endpoint — dev-only, never exposed in production (rules.md §3)
if ENABLE_ADMIN:
    from fastapi import APIRouter

    admin_router = APIRouter(prefix="/admin", tags=["Admin"])

    @admin_router.post("/reingest")
    async def reingest():
        """Trigger the ingestion pipeline manually. Dev-only."""
        from src.ingestion.embedder import run_ingestion

        count = run_ingestion()
        return {"status": "ok", "chunks_ingested": count}

    app.include_router(admin_router)


if __name__ == "__main__":
    import uvicorn
    from src.config import API_HOST, API_PORT

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host=API_HOST, port=API_PORT)
