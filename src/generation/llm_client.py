"""
Backend-agnostic LLM client — the single switch between dev and prod inference.

Why two backends behind one interface: architecture.md §2.6 explains this is the
most important architecture decision. Free cloud hosting can't run local LLMs,
so we switch between Ollama (dev, free, local) and Groq (prod, free API) via
one config value. This dev/prod split is a legitimate real-world pattern.
"""

from __future__ import annotations

import logging

from src.config import LLM_BACKEND

logger = logging.getLogger(__name__)


def generate(system_prompt: str, user_prompt: str) -> str:
    """
    Route generation to the configured backend (Ollama or Groq).

    The backend is selected by the LLM_BACKEND env variable — no code changes
    needed to switch between local dev and production (FR-7).
    """
    backend = LLM_BACKEND.lower()

    if backend == "ollama":
        from src.generation import ollama_backend

        logger.debug("Using Ollama backend")
        return ollama_backend.generate(system_prompt, user_prompt)

    elif backend == "groq":
        from src.generation import groq_backend

        logger.debug("Using Groq backend")
        return groq_backend.generate(system_prompt, user_prompt)

    else:
        raise ValueError(
            f"Unknown LLM_BACKEND: '{backend}'. Must be 'ollama' or 'groq'. "
            "Set in .env file."
        )


def get_active_backend() -> str:
    """Return the name of the currently configured backend."""
    return LLM_BACKEND.lower()
