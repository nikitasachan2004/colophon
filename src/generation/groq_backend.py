"""
Groq backend — hosted LLM inference for production.

Why: free-tier API serving open-weight models on fast inference hardware.
Groq's free tier requires no credit card and gives ~14,400 requests/day
and ~30 requests/minute on llama-3.1-8b-instant — comfortably enough for
a portfolio demo (architecture.md §2.6).
"""

from __future__ import annotations

import logging

from groq import Groq

from src.config import GROQ_API_KEY, GROQ_MODEL

logger = logging.getLogger(__name__)


def generate(
    system_prompt: str,
    user_prompt: str,
    model: str | None = None,
) -> str:
    """
    Generate a response using the Groq hosted API.

    Args:
        system_prompt: System instructions for the model.
        user_prompt: The user's question with context.
        model: Override for the Groq model name.

    Returns:
        The generated text response.

    Raises:
        ValueError: If GROQ_API_KEY is not configured.
        RuntimeError: If the API call fails.
    """
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is not set. Get a free key at https://console.groq.com "
            "and add it to your .env file."
        )

    model = model or GROQ_MODEL
    client = Groq(api_key=GROQ_API_KEY)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,  # Low temperature for factual, grounded answers
            max_tokens=1024,
        )
    except Exception as exc:
        raise RuntimeError(f"Groq API error: {exc}")

    return response.choices[0].message.content or ""
