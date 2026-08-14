"""
Ollama backend — local LLM inference for development.

Why: genuinely free, fully private, good for iterating on prompts/retrieval
without any API limits. Uses Ollama's HTTP API directly (architecture.md §2.6).
"""

from __future__ import annotations

import logging

import httpx

from src.config import OLLAMA_BASE_URL, OLLAMA_MODEL

logger = logging.getLogger(__name__)


def generate(
    system_prompt: str,
    user_prompt: str,
    model: str | None = None,
) -> str:
    """
    Generate a response using the local Ollama instance.

    Args:
        system_prompt: System instructions for the model.
        user_prompt: The user's question with context.
        model: Override for the Ollama model name.

    Returns:
        The generated text response.

    Raises:
        ConnectionError: If Ollama is not running.
        RuntimeError: If the API call fails.
    """
    model = model or OLLAMA_MODEL
    url = f"{OLLAMA_BASE_URL}/api/chat"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
    }

    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
    except httpx.ConnectError:
        raise ConnectionError(
            f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. "
            "Is Ollama running? Start with: ollama serve"
        )
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(f"Ollama API error: {exc.response.status_code} — {exc.response.text}")

    data = resp.json()
    return data.get("message", {}).get("content", "")
