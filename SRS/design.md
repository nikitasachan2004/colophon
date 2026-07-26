# Design Document
## Production RAG Knowledge Assistant

This document is the detailed, implementation-level companion to `architecture.md`. Where architecture.md answers "what components and why," this answers "exactly how each piece is built" — data models, algorithms, API contracts, prompts, folder layout. Anyone (including an AI coding assistant) should be able to implement directly from this file.

---

## 1. Repository Structure

```
rag-knowledge-assistant/
├── src/
│   ├── ingestion/
│   │   ├── scraper.py          # fetch + clean HTML → structured markdown
│   │   ├── chunker.py          # structure-aware chunking with overlap
│   │   └── embedder.py         # embed chunks, write to Chroma
│   ├── retrieval/
│   │   ├── vector_search.py    # Chroma dense search
│   │   ├── bm25_search.py      # sparse keyword search
│   │   ├── fusion.py           # Reciprocal Rank Fusion
│   │   └── reranker.py         # cross-encoder reranking
│   ├── generation/
│   │   ├── llm_client.py       # backend-agnostic interface
│   │   ├── ollama_backend.py
│   │   ├── groq_backend.py
│   │   └── prompts.py          # system + user prompt templates
│   ├── api/
│   │   ├── main.py             # FastAPI app
│   │   ├── schemas.py          # Pydantic request/response models
│   │   └── routes/
│   │       ├── query.py
│   │       └── health.py
│   ├── evaluation/
│   │   ├── eval_questions.json # held-out question set
│   │   ├── run_eval.py         # runs full pipeline, computes RAGAS
│   │   └── results/            # timestamped results.json history
│   └── config.py               # all env-driven config, single source of truth
├── frontend/
│   └── app.py                  # Streamlit UI
├── tests/
│   ├── unit/
│   │   ├── test_chunker.py
│   │   ├── test_fusion.py
│   │   └── test_reranker.py
│   └── integration/
│       ├── test_query_pipeline.py
│       └── test_no_context_found.py
├── data/
│   ├── raw/                    # scraped raw HTML (gitignored, cached)
│   ├── processed/               # cleaned chunks (gitignored)
│   └── chroma_db/               # persisted vector store (gitignored)
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.frontend
│   └── docker-compose.yml
├── .github/workflows/ci.yml
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
└── ARCHITECTURE.md / DESIGN.md / etc. (this doc set, versioned in-repo)
```

---

## 2. Data Models

### 2.1 Chunk (stored in ChromaDB, with metadata)
```python
{
  "id": "anthropic-api-messages-create-chunk-3",
  "text": "To create a Message, call the Messages API...",
  "embedding": [0.0123, -0.0456, ...],       # 384-dim, all-MiniLM-L6-v2
  "metadata": {
    "source_url": "https://docs.claude.com/en/api/messages",
    "section_heading": "Create a Message",
    "doc_category": "messages_api",           # coarse category for optional filtering (FR-11 stretch)
    "chunk_index": 3,
    "token_count": 487,
    "ingested_at": "2026-09-01T12:00:00Z"
  }
}
```

### 2.2 Query Request (API contract)
```python
class QueryRequest(BaseModel):
    question: str
    top_k: int = 5              # final number of chunks passed to LLM after rerank
    doc_category: str | None = None   # optional stretch filter (FR-11)
```

### 2.3 Query Response
```python
class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]   # what was actually used to ground the answer
    latency_ms: int
    backend_used: Literal["ollama", "groq"]
    context_found: bool          # false triggers the "I don't know" path

class SourceChunk(BaseModel):
    text: str
    source_url: str
    section_heading: str
    relevance_score: float       # post-rerank score
```

### 2.4 Evaluation Question (held-out set)
```python
{
  "id": "eval-001",
  "question": "How do I make Claude use a specific tool instead of choosing automatically?",
  "expected_source_url": "https://docs.claude.com/en/docs/agents-and-tools/tool-use/overview",
  "category": "tool_use",
  "notes": "Tests exact-match retrieval of a specific API parameter (tool_choice)"
}
```
Held-out set target size: **25–40 questions**, hand-written by you after reading the actual docs (not generated blindly), covering: (a) direct factual lookups, (b) questions phrased differently from the doc's exact wording (tests semantic retrieval), (c) at least 5 questions **deliberately outside the corpus** (tests the "I don't know" path — FR-10).

---

## 3. Chunking Algorithm (structure-aware, the core differentiator)

```
function chunk_document(markdown_text, source_url):
    sections = split_on_headings(markdown_text)   # split on ##, ### boundaries
    chunks = []
    for section in sections:
        if token_count(section) <= MAX_CHUNK_TOKENS (target: 512):
            chunks.append(section as one chunk, tagged with its heading)
        else:
            # sub-split oversized sections using recursive character/token
            # splitting, preserving code blocks as atomic units (never split
            # a code block mid-way — this is an API docs-specific rule)
            sub_chunks = recursive_split(section, max_tokens=512, overlap=0.10-0.15)
            chunks.append(each sub_chunk, tagged with parent heading)
    return chunks
```

**Explicit rules:**
- Never split inside a fenced code block (```...```) — API doc code examples must stay atomic, or retrieval returns broken/unusable snippets.
- Overlap of 10–15% between adjacent sub-chunks within the same section, so a sentence spanning a chunk boundary isn't lost.
- Every chunk carries its parent heading as metadata, even after sub-splitting — this is what makes "section_heading" in the data model meaningful and improves citation quality.
- Chunk size (512 tokens) and overlap (12%) are **named constants in `config.py`**, not magic numbers — they will be tuned as a hyperparameter in Phase 4 against RAGAS scores, and every value tried is logged in `memory.md`.

---

## 4. Retrieval & Fusion Algorithm

```
function hybrid_retrieve(query, k=20):
    dense_results = chroma_search(embed(query), top_k=k)      # cosine similarity
    sparse_results = bm25_search(query, top_k=k)               # keyword scoring
    fused = reciprocal_rank_fusion(dense_results, sparse_results, k_constant=60)
    return fused  # ranked list, deduplicated by chunk id

function reciprocal_rank_fusion(list_a, list_b, k_constant):
    scores = {}
    for rank, item in enumerate(list_a):
        scores[item.id] += 1 / (k_constant + rank)
    for rank, item in enumerate(list_b):
        scores[item.id] += 1 / (k_constant + rank)
    return sorted(scores, descending)
```

Then:
```
function rerank(query, candidates, final_k=5):
    scored = [(c, cross_encoder_score(query, c.text)) for c in candidates]
    top = sorted(scored, by score, descending)[:final_k]
    return top
```

**"No context found" rule (FR-10):** if the top reranked chunk's score is below a defined confidence threshold (tuned empirically in Phase 4, start at a conservative value and adjust against the eval set's "outside corpus" questions), the system sets `context_found = false` and the LLM is prompted with an explicit instruction to say it doesn't have enough information, rather than being sent weak context and asked to "do its best."

---

## 5. Prompt Templates

### 5.1 System Prompt (fixed)
```
You are a documentation assistant for [COMPANY] API docs. You must answer
ONLY using the provided context chunks below. Do not use any knowledge
outside of the provided context, even if you believe you know the answer.

If the provided context does not contain enough information to answer the
question, respond exactly: "I don't have enough information in the
documentation to answer that confidently." Do not guess.

When you answer, cite which source section(s) you used.
```

### 5.2 User Prompt Template
```
Context:
[1] (Source: {section_heading_1}, {source_url_1})
{chunk_text_1}

[2] (Source: {section_heading_2}, {source_url_2})
{chunk_text_2}

... (up to top-5 reranked chunks)

Question: {user_question}

Answer using only the context above, and cite sources by number.
```

Rationale: numbered citations in the prompt make it easy to parse the model's citation references back into the `sources` field of the API response programmatically, rather than relying on fuzzy string matching.

---

## 6. Evaluation Design

- **Metrics:** faithfulness (is the answer's content actually supported by the retrieved context — the core hallucination check), answer relevancy (does the answer address the question asked), context precision (are the retrieved chunks actually relevant), context recall (did retrieval find the chunk(s) that contain the real answer, measured against `expected_source_url`).
- **Judge model:** Groq-hosted `llama-3.1-8b-instant` (or `llama-3.3-70b-versatile` if the free-tier rate limit allows for the smaller eval set size) used as the RAGAS judge — keeps evaluation fully free.
- **Process:** `run_eval.py` runs every held-out question through the *exact same* production query pipeline (not a shortcut), logs raw retrieval + generation output, computes RAGAS scores, and writes a timestamped JSON to `evaluation/results/`. This produces a **history**, not a single snapshot — so the README's "before/after" claim (e.g., after tuning chunk size or adding reranking) is backed by two real, dated runs.
- **Baseline run:** naive fixed-size chunking, vector-only retrieval, no reranking — deliberately the "worst reasonable version," run once at the start of Phase 4 to establish the honest starting point.
- **Iteration:** each subsequent change (structure-aware chunking → hybrid retrieval → reranking) is measured independently against the same question set, so the README can honestly attribute the improvement to a specific change, not just report one final number.

---

## 7. Configuration (`config.py` / `.env`)

```
# .env.example
LLM_BACKEND=ollama              # "ollama" | "groq" — the single switch described in architecture.md §2.6
OLLAMA_MODEL=llama3.1:8b
GROQ_API_KEY=                   # required only if LLM_BACKEND=groq
GROQ_MODEL=llama-3.1-8b-instant
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHUNK_SIZE_TOKENS=512
CHUNK_OVERLAP_PCT=0.12
RETRIEVAL_TOP_K=20
RERANK_FINAL_K=5
CONFIDENCE_THRESHOLD=0.35       # tuned in Phase 4 against eval set
CHROMA_PERSIST_DIR=./data/chroma_db
```

No secret ever hardcoded in source — enforced by `rules.md` §3.

---

## 8. Frontend Design (Streamlit) — minimal but complete

- Text input for the question
- "Ask" button → calls `POST /query`
- Display: answer text, then an expandable "Sources" section listing each cited chunk with its doc link and relevance score
- Small footer showing latency and which backend answered (Ollama/Groq) — useful for your own debugging and a nice transparency touch for anyone reviewing the demo
- A visible disclaimer: "Answers are grounded in [COMPANY]'s public docs as of [ingestion date]; this is an independent portfolio project, not affiliated with [COMPANY]."

---

## 9. API Contract Summary

```
POST /query
  body: QueryRequest
  returns: QueryResponse

GET /health
  returns: {"status": "ok", "backend": "groq", "chunks_indexed": 1842}

POST /admin/reingest   (dev-only, not exposed in prod deployment)
  triggers the ingestion pipeline manually
```
