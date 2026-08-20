"""
Health check route — GET /health.

Returns service status, active LLM backend, and count of indexed chunks.
Provides a quick way to verify the system is running and has data (design.md §9).
"""

from __future__ import annotations

from fastapi import APIRouter

from src.api.schemas import HealthResponse
from src.generation.llm_client import get_active_backend
from src.ingestion.embedder import get_collection

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Service health check with backend info and chunk count."""
    try:
        collection = get_collection()
        chunks_indexed = collection.count()
    except Exception:
        chunks_indexed = 0

    return HealthResponse(
        status="ok",
        backend=get_active_backend(),
        chunks_indexed=chunks_indexed,
    )
