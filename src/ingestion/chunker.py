"""
Structure-aware chunker — the core differentiator of this RAG system.

Why structure-aware over fixed-size: 2026 benchmarks show chunking strategy is the
single highest-leverage lever in RAG quality (+20-74pp retrieval accuracy swings).
API docs have strong heading structure that we exploit: split on headings first,
then sub-split oversized sections with overlap.

Key rules (design.md §3):
- Never split inside a fenced code block — API code examples must stay atomic.
- 10-15% overlap between adjacent sub-chunks within the same section.
- Every chunk carries its parent heading as metadata.
- Chunk size and overlap are configurable (config.py), not magic numbers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import tiktoken

from src.config import CHUNK_OVERLAP_PCT, CHUNK_SIZE_TOKENS

_TOKENIZER = None


def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        try:
            import tiktoken

            _TOKENIZER = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _TOKENIZER = "fallback"
    return _TOKENIZER


@dataclass
class Chunk:
    """A single chunk of documentation text with metadata."""

    text: str
    source_url: str
    section_heading: str
    doc_category: str
    chunk_index: int
    token_count: int = field(default=0)

    def __post_init__(self) -> None:
        if self.token_count == 0:
            self.token_count = count_tokens(self.text)


def count_tokens(text: str) -> int:
    """Count tokens using the cl100k_base tokenizer (falling back to word count if offline)."""
    tok = _get_tokenizer()
    if tok == "fallback" or tok is None:
        return len(text.split())
    try:
        return len(tok.encode(text))
    except Exception:
        return len(text.split())


def _split_on_headings(markdown: str) -> list[tuple[str, str]]:
    """
    Split markdown into (heading, body) pairs at ## and ### boundaries.
    Returns a list of (heading_text, section_body) tuples.
    """
    # Match ## and ### headings (not # — that's the page title, kept with first section)
    heading_pattern = re.compile(r"^(#{2,3})\s+(.+)$", re.MULTILINE)

    sections: list[tuple[str, str]] = []
    matches = list(heading_pattern.finditer(markdown))

    if not matches:
        # No headings found — treat entire text as one section
        return [("", markdown.strip())]

    # Content before first heading
    preamble = markdown[: matches[0].start()].strip()
    if preamble:
        # Strip the source comment if present
        cleaned = re.sub(r"^<!--.*?-->\s*", "", preamble, flags=re.DOTALL).strip()
        if cleaned:
            sections.append(("Introduction", cleaned))

    for i, match in enumerate(matches):
        heading = match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        body = markdown[start:end].strip()
        if body:
            sections.append((heading, body))

    return sections


def _extract_code_blocks(text: str) -> list[tuple[int, int]]:
    """Find all fenced code block spans (start, end) in the text."""
    spans: list[tuple[int, int]] = []
    for match in re.finditer(r"```[\s\S]*?```", text):
        spans.append((match.start(), match.end()))
    return spans


def _is_inside_code_block(pos: int, code_spans: list[tuple[int, int]]) -> bool:
    """Check if a position falls inside any code block."""
    return any(start <= pos < end for start, end in code_spans)


def _split_paragraphs_preserving_code_blocks(text: str) -> list[str]:
    """Split on blank lines, but keep fenced code blocks whole."""
    parts: list[str] = []
    current: list[str] = []
    in_code_block = False

    for line in text.splitlines():
        current.append(line)
        if line.lstrip().startswith("```"):
            in_code_block = not in_code_block
        if not in_code_block and not line.strip():
            part = "\n".join(current).strip()
            if part:
                parts.append(part)
            current = []

    tail = "\n".join(current).strip()
    if tail:
        parts.append(tail)

    return parts


def _recursive_split(
    text: str,
    max_tokens: int,
    overlap_pct: float,
) -> list[str]:
    """
    Sub-split oversized text into smaller chunks with overlap.
    Never splits inside fenced code blocks.
    """
    if count_tokens(text) <= max_tokens:
        return [text]

    overlap_tokens = int(max_tokens * overlap_pct)

    # Split on paragraph boundaries (double newlines) first
    paragraphs = _split_paragraphs_preserving_code_blocks(text)

    chunks: list[str] = []
    current_parts: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = count_tokens(para)

        # If a single paragraph (e.g., a code block) exceeds max_tokens,
        # keep it whole — never split code blocks
        if para_tokens > max_tokens and para.strip().startswith("```"):
            # Flush current buffer
            if current_parts:
                chunks.append("\n\n".join(current_parts))
                current_parts = []
                current_tokens = 0
            # Code block goes as its own chunk (even if oversized)
            chunks.append(para)
            continue

        if current_tokens + para_tokens > max_tokens and current_parts:
            # Flush current chunk
            chunks.append("\n\n".join(current_parts))

            # Overlap: keep the last portion of text
            overlap_text: list[str] = []
            overlap_count = 0
            for prev_part in reversed(current_parts):
                prev_tokens = count_tokens(prev_part)
                if overlap_count + prev_tokens > overlap_tokens:
                    break
                overlap_text.insert(0, prev_part)
                overlap_count += prev_tokens

            current_parts = overlap_text
            current_tokens = overlap_count

        current_parts.append(para)
        current_tokens += para_tokens

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return chunks


def chunk_document(
    markdown: str,
    source_url: str,
    doc_category: str,
    max_tokens: int | None = None,
    overlap_pct: float | None = None,
) -> list[Chunk]:
    """
    Structure-aware chunking per design.md §3.

    1. Split on heading boundaries (## / ###).
    2. If a section fits within max_tokens, keep it as one chunk.
    3. If oversized, sub-split with overlap — but never split inside code blocks.
    4. Every chunk carries its parent heading as metadata.

    Args:
        markdown: Cleaned markdown text of a documentation page.
        source_url: The original URL this content was scraped from.
        doc_category: Coarse category (messages_api, tools_and_agents, mcp, etc.).
        max_tokens: Override for CHUNK_SIZE_TOKENS.
        overlap_pct: Override for CHUNK_OVERLAP_PCT.

    Returns:
        List of Chunk objects with metadata.
    """
    if not markdown or not markdown.strip():
        return []

    max_tokens = max_tokens or CHUNK_SIZE_TOKENS
    overlap_pct = overlap_pct or CHUNK_OVERLAP_PCT

    sections = _split_on_headings(markdown)
    chunks: list[Chunk] = []
    chunk_index = 0

    for heading, body in sections:
        body_tokens = count_tokens(body)

        if body_tokens <= max_tokens:
            # Section fits — one chunk
            chunks.append(
                Chunk(
                    text=body,
                    source_url=source_url,
                    section_heading=heading,
                    doc_category=doc_category,
                    chunk_index=chunk_index,
                )
            )
            chunk_index += 1
        else:
            # Sub-split oversized sections
            sub_chunks = _recursive_split(body, max_tokens, overlap_pct)
            for sub_text in sub_chunks:
                sub_text = sub_text.strip()
                if sub_text:
                    chunks.append(
                        Chunk(
                            text=sub_text,
                            source_url=source_url,
                            section_heading=heading,
                            doc_category=doc_category,
                            chunk_index=chunk_index,
                        )
                    )
                    chunk_index += 1

    return chunks


def chunk_documents(documents: list[dict[str, str]]) -> list[Chunk]:
    """
    Chunk all scraped documents. Convenience wrapper over chunk_document.

    Args:
        documents: List of dicts with keys "url", "markdown", "category"
                   (output of scraper.scrape_all).

    Returns:
        Flat list of all Chunk objects across all documents.
    """
    all_chunks: list[Chunk] = []
    for doc in documents:
        doc_chunks = chunk_document(
            markdown=doc["markdown"],
            source_url=doc["url"],
            doc_category=doc["category"],
        )
        all_chunks.extend(doc_chunks)
    return all_chunks


if __name__ == "__main__":
    # Quick test with a synthetic doc
    test_md = """
## Make a Messages API request

To create a message, call `client.messages.create`.

```python
import anthropic

client = anthropic.Anthropic()
message = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello, Claude"}],
)
```

### Parameters

The following parameters are required:

- `model` — the Claude model to use
- `max_tokens` — the maximum number of tokens to generate
- `messages` — the conversation messages to send

### Returns

Returns a Message object on success.

## Handle Tool Calls

Use `tool_use` blocks to run client-side tools and return `tool_result` blocks.
"""

    chunks = chunk_document(
        test_md,
        source_url="https://platform.claude.com/docs/en/build-with-claude/working-with-messages",
        doc_category="messages_api",
    )
    for c in chunks:
        print(f"[{c.section_heading}] ({c.token_count} tokens) {c.text[:80]}...")
