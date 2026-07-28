"""
Integration tests for the FastAPI query pipeline.

Validates:
1. /health endpoint response
2. /query endpoint request/response format
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from src.api.main import app
from src.retrieval.vector_search import SearchResult

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "backend" in data
    assert "chunks_indexed" in data


@patch("src.api.routes.query.llm_client.generate")
@patch("src.api.routes.query.vector_search")
@patch("src.api.routes.query.bm25_search")
@patch("src.api.routes.query.rerank")
def test_query_pipeline_success(mock_rerank, mock_bm25, mock_vector, mock_llm):
    # Mock retrieval & reranking hits
    sample_result = SearchResult(
        "c1",
        "Use client.messages.create with model and messages",
        0.95,
        "https://platform.claude.com/docs/en/api",
        "Messages",
        "messages_api",
    )
    mock_vector.return_value = [sample_result]
    mock_bm25.return_value = [sample_result]
    mock_rerank.return_value = ([sample_result], True)
    mock_llm.return_value = "To create a message, call client.messages.create [1]."

    payload = {"question": "How do I create a message in Claude Messages API?", "top_k": 3}
    response = client.post("/query", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert data["context_found"] is True
    assert len(data["sources"]) > 0
    assert data["sources"][0]["section_heading"] == "Messages"
    assert data["latency_ms"] >= 0
