"""
Pydantic request/response models for the FastAPI endpoints.

Matches the API contract in design.md §2.2, §2.3, §9.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request body for POST /query."""

    question: str = Field(..., description="Natural-language question about Anthropic Claude API docs")
    top_k: int = Field(default=5, description="Final number of chunks passed to LLM after rerank")
    doc_category: str | None = Field(default=None, description="Optional category filter (FR-11 stretch)")


class SourceChunk(BaseModel):
    """A single source chunk that grounded the answer."""

    text: str
    source_url: str
    section_heading: str
    relevance_score: float = Field(description="Post-rerank relevance score")


class QueryResponse(BaseModel):
    """Response body for POST /query."""

    answer: str
    sources: list[SourceChunk]
    latency_ms: int
    backend_used: str  # "ollama" or "groq"
    context_found: bool = Field(description="False triggers the 'I don't know' path")


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str = "ok"
    backend: str
    chunks_indexed: int
