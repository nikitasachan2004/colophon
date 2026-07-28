"""
Integration test for the 'no context found' out-of-domain rejection path (FR-10).

Validates:
When retrieval returns zero or below-threshold confidence matches, the system:
1. Sets context_found = False
2. Returns the strict refusal message without hallucinating
3. Omits invalid source citations
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from src.api.main import app
from src.generation.prompts import NO_CONTEXT_RESPONSE

client = TestClient(app)


@patch("src.api.routes.query.vector_search")
@patch("src.api.routes.query.bm25_search")
@patch("src.api.routes.query.rerank")
def test_out_of_domain_question_refusal(mock_rerank, mock_bm25, mock_vector):
    # Mock no relevant context found
    mock_vector.return_value = []
    mock_bm25.return_value = []
    mock_rerank.return_value = ([], False)

    payload = {
        "question": "What is the secret recipe for Martian ice cream on interstellar flights?",
        "top_k": 5,
    }
    response = client.post("/query", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["context_found"] is False
    assert data["answer"] == NO_CONTEXT_RESPONSE
    assert len(data["sources"]) == 0
