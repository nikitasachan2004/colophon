"""
Unit tests for the structure-aware chunker.

Validates:
1. Heading boundary splitting and metadata preservation
2. Atomic code block preservation (never splitting inside fenced blocks)
3. Sub-splitting oversized sections with overlap
"""

import pytest
from src.ingestion.chunker import (
    Chunk,
    _split_on_headings,
    _recursive_split,
    chunk_document,
    count_tokens,
)


def test_split_on_headings_standard():
    md = """
## Tool Use Overview
This is the body describing tool use.

### Defining Tools
This is subsection content on tool definitions.

## Handling Tool Calls
Final section content.
"""
    sections = _split_on_headings(md)
    assert len(sections) == 3
    assert sections[0][0] == "Tool Use Overview"
    assert "This is the body describing tool use." in sections[0][1]
    assert sections[1][0] == "Defining Tools"
    assert sections[2][0] == "Handling Tool Calls"


def test_code_block_remains_atomic():
    code_block = (
        "```python\n"
        + "\n".join([f"param_{i} = {i}" for i in range(100)])
        + "\n```"
    )
    md = f"""
## Code Example
Here is some introductory text before the long code snippet.

{code_block}

Closing summary after code block.
"""
    chunks = chunk_document(
        markdown=md,
        source_url="https://platform.claude.com/docs/en/api/test",
        doc_category="test",
        max_tokens=60,  # Deliberately small to force splitting
        overlap_pct=0.1,
    )

    # Check that the code block was not bisected/broken into invalid code
    code_chunks = [c for c in chunks if "```python" in c.text]
    assert len(code_chunks) >= 1
    for c in code_chunks:
        assert c.text.count("```") % 2 == 0


def test_code_block_with_blank_lines_remains_atomic():
    code_block = """```python
import anthropic

client = anthropic.Anthropic()

message = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}],
)
```"""
    md = f"""
## Messages example
Short intro.

{code_block}

Closing text that should split separately if needed.
"""
    chunks = chunk_document(
        markdown=md,
        source_url="https://docs.claude.com/en/api/messages/create",
        doc_category="messages_api",
        max_tokens=30,
        overlap_pct=0.1,
    )

    code_chunks = [c for c in chunks if "client.messages.create" in c.text]
    assert len(code_chunks) == 1
    assert code_chunks[0].text.count("```") == 2


def test_heading_metadata_propagation():
    md = """
## Authentication & Security
Paragraph 1 about API keys and headers.

Paragraph 2 about rate limits and token headers.

Paragraph 3 about workspace secret management.
"""
    chunks = chunk_document(
        markdown=md,
        source_url="https://platform.claude.com/docs/en/manage-claude/authentication",
        doc_category="management",
        max_tokens=15,  # Force sub-splitting into multiple chunks
        overlap_pct=0.1,
    )

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.section_heading == "Authentication & Security"
        assert chunk.doc_category == "management"
        assert chunk.source_url == "https://platform.claude.com/docs/en/manage-claude/authentication"
        assert chunk.token_count > 0


def test_empty_document():
    chunks = chunk_document(
        markdown="",
        source_url="https://platform.claude.com/docs/en/empty",
        doc_category="general",
    )
    assert chunks == []
