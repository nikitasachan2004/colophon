"""
Prompt templates for the RAG generation pipeline.

Why these specific templates: design.md §5 specifies a strict system prompt that
forces the model to answer only from provided context and say "I don't know"
when context is insufficient. Numbered citations enable programmatic parsing
of source references back into the API response.
"""

from __future__ import annotations

SYSTEM_PROMPT = """You are a documentation assistant for Anthropic's Claude API documentation. You must answer \
ONLY using the provided context chunks below. Do not use any knowledge \
outside of the provided context, even if you believe you know the answer.

If the provided context does not contain enough information to answer the \
question, respond exactly: "I don't have enough information in the \
documentation to answer that confidently." Do not guess.

When you answer, cite which source section(s) you used by referencing them \
by their number (e.g., [1], [2])."""


NO_CONTEXT_RESPONSE = (
    "I don't have enough information in the documentation to answer that confidently."
)


def build_user_prompt(
    question: str,
    chunks: list[dict[str, str]],
) -> str:
    """
    Build the user prompt with numbered context chunks.

    Args:
        question: The user's natural-language question.
        chunks: List of dicts with keys "text", "section_heading", "source_url".

    Returns:
        Formatted user prompt string.
    """
    context_parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        heading = chunk.get("section_heading", "Unknown section")
        url = chunk.get("source_url", "")
        text = chunk.get("text", "")
        context_parts.append(f"[{i}] (Source: {heading}, {url})\n{text}")

    context_block = "\n\n".join(context_parts)

    return f"""Context:
{context_block}

Question: {question}

Answer using only the context above, and cite sources by number."""
