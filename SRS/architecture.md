# Architecture Document
## Production RAG Knowledge Assistant

This document describes the *how*: components, data flow, and — critically — the reasoning behind every tech choice, so you can defend each one in an interview. Every choice below is justified against 2026 production-RAG practice (chunking, hybrid retrieval, reranking) and against our hard constraint of $0 budget.

---

## 1. High-Level System Diagram

```
                         ┌─────────────────────────────────────────┐
                         │            INGESTION PIPELINE            │
                         │              (offline / batch)            │
                         │                                            │
   Doc source URLs ───▶  │  Scraper → Cleaner → Structure-Aware      │
                         │  Chunker → Embedder → Vector Store Writer │
                         │                                            │
                         └──────────────────────┬─────────────────────┘
                                                  │
                                                  ▼
                                     ┌─────────────────────┐
                                     │   ChromaDB (vector)   │
                                     │  + BM25 index (sparse)│
                                     └───────────┬───────────┘
                                                  │
        ┌─────────────────────────────────────────┼─────────────────────────────────┐
        │                                QUERY-TIME PIPELINE (online)                 │
        │                                                                              │
User ──▶│  FastAPI /query endpoint                                                     │
        │       │                                                                      │
        │       ▼                                                                      │
        │  1. Hybrid Retrieval (vector top-20 ⊕ BM25 top-20 → RRF fusion)               │
        │       │                                                                      │
        │       ▼                                                                      │
        │  2. Reranking (cross-encoder, top-20 → top-5)                                 │
        │       │                                                                      │
        │       ▼                                                                      │
        │  3. Prompt Assembly (top-5 chunks + citation metadata + system prompt)         │
        │       │                                                                      │
        │       ▼                                                                      │
        │  4. LLM Generation  ── config-switched backend ──▶  Ollama (dev)  /  Groq (prod)│
        │       │                                                                      │
        │       ▼                                                                      │
        │  5. Response + cited sources + latency  ──▶  Streamlit frontend               │
        └──────────────────────────────────────────────────────────────────────────────┘

                         ┌─────────────────────────────────────────┐
                         │           EVALUATION PIPELINE             │
                         │             (offline / on-demand)          │
                         │                                            │
                         │  Held-out question set → run through the  │
                         │  same query pipeline → RAGAS scoring       │
                         │  (faithfulness, answer relevancy,          │
                         │   context precision/recall) → results.json │
                         └─────────────────────────────────────────┘
```

---

## 2. Component-by-Component Breakdown & Justification

### 2.1 Ingestion Pipeline (offline)
**What it does:** Fetches documentation pages, strips navigation/boilerplate HTML, splits into chunks aligned with document structure, embeds each chunk, writes to the vector store.

**Why structure-aware chunking, not fixed-size:** 2026 benchmarking shows chunking strategy is the single highest-leverage lever in RAG quality — changing chunking strategy alone (same model, same everything else) has been shown to swing retrieval accuracy by +20–74 percentage points. Fixed-size chunking is described as a "fast start and a slow ceiling" — fine for an MVP, but API documentation has strong heading structure (`##`/`###` sections, code blocks) that we should exploit: split on headings first, then sub-split any section that's still too long, with 10–15% overlap between chunks so a fact split across two chunks isn't lost. This is the documented "basic production recipe" for 2026 RAG systems.

**Why this specific target chunk size:** ~512 tokens is the pragmatic default per a February 2026 multi-strategy benchmark, though we treat this as a tunable hyperparameter (see `design.md` §3) rather than a fixed constant — we will measure, not assume.

### 2.2 Vector Store — ChromaDB
**Why Chroma over Pinecone/Weaviate/Qdrant:** Free, runs embedded/local with zero external account or network dependency, persists to disk, and is exactly what appears in the majority of 2026 "AI engineer portfolio project" references as the standard free/local choice. Trade-off acknowledged: Chroma doesn't scale to enterprise volumes the way Pinecone does — this is a deliberate, documented choice appropriate to project scale, and the README will say so explicitly (this is itself a good "trade-off awareness" talking point for interviews).

### 2.3 Sparse Retrieval — BM25 (via `rank_bm25` or Chroma's built-in hybrid support)
**Why hybrid, not vector-only:** Vector search is good at "meaning" but can miss exact-match terms — critical for API docs where a user might type an exact method or parameter name (`client.messages.create`, `tool_choice`) that BM25's keyword matching catches reliably and dense embeddings sometimes blur. 2026 best-practice guides list hybrid search + reciprocal rank fusion as one of the four highest-leverage levers in production RAG, alongside chunking and reranking.

### 2.4 Reranker — Cross-encoder (BGE-reranker-v2, open-source)
**Why we add a reranking stage:** Retrieval (bi-encoder / BM25) is cheap and fast but only "topically similar" — a cross-encoder reranker looks at the query and each candidate chunk *together*, which is described as "often the difference between topically similar and actually answers the question." Standard 2026 recipe: retrieve top-20 candidates cheaply, rerank down to top 3–5 before they reach the LLM, which also avoids the "lost in the middle" problem where long-context LLMs quietly ignore information buried deep in a large context window.
**Why BGE-reranker-v2 specifically:** Open-source, free, and small enough to run on a single consumer GPU or even CPU (slower) — unlike Cohere Rerank or Voyage rerank, which are paid APIs. This keeps us at $0.

### 2.5 Embeddings — `sentence-transformers/all-MiniLM-L6-v2`
**Why this over Ollama's embedding models or an API:** Runs entirely on CPU, no GPU or API key required, widely used, small (~80MB), and removes one moving part (we don't need Ollama running just to embed — only for generation in dev mode). This is a deliberate simplicity choice for a $0/solo-dev project.

### 2.6 Generation — Dual Backend: Ollama (dev) + Groq (prod)
**Why two backends instead of one:** This is the most important architecture decision in the whole project, and it should be stated explicitly, not hidden. Free cloud hosting tiers (Render, Railway, Fly.io, Hugging Face Spaces free CPU tier) cannot run a local LLM like Ollama — no GPU, insufficient RAM, and free containers frequently sleep/restart, which would drop a loaded model. So:
- **Local development** uses Ollama running Llama 3.1 8B or Qwen2.5 7B — genuinely free, fully private, good for iterating on prompts/retrieval without any API limits.
- **Public deployment** switches (via one config value, see `design.md` §7) to **Groq's free API**, serving `llama-3.1-8b-instant` — also an open-weight model, just hosted on Groq's fast inference hardware. Groq's free tier requires no credit card and gives roughly 14,400 requests/day and ~30 requests/minute on this model, comfortably enough for a portfolio demo's traffic.
This dev/prod split is a legitimate, common real-world pattern (most companies don't run inference on their own web server either) and is a strong interview talking point about environment-aware design.

### 2.7 Backend API — FastAPI
**Why FastAPI over Flask/Django:** Async support (matters for I/O-bound LLM/embedding calls), automatic interactive API docs (`/docs`) which itself demonstrates production polish, and it's the dominant choice across 2026 AI-engineer project references.

### 2.8 Frontend — Streamlit
**Why Streamlit over a full React app:** Given the 60–80 hour budget, the differentiating engineering work is in retrieval quality and evaluation, not frontend polish. Streamlit lets us ship a clean, functional UI in hours, not days, and deploys natively to Hugging Face Spaces. (If time remains at the end, a small React frontend is a documented stretch goal — not the default plan.)

### 2.9 Evaluation — RAGAS
**Why RAGAS specifically:** It's the most-referenced open-source RAG evaluation framework in 2026 sources, computes the exact metrics that matter for this project (faithfulness — is the answer actually grounded in retrieved context; answer relevancy; context precision/recall), and can use Groq as the free "judge" LLM instead of requiring a paid OpenAI judge model.

### 2.10 Deployment — Hugging Face Spaces (Docker SDK)
**Why HF Spaces over Render/Railway/Fly.io free tiers:** Recognized by AI-focused recruiters as the standard place to host exactly this kind of project, supports full custom Docker containers (not just simple web apps), always-on public URL, and genuinely free with no time-boxed trial.

### 2.11 CI — GitHub Actions
**Why:** Free for public repos, and a passing CI badge on the README is a small but real signal of engineering discipline that many student projects skip entirely.

---

## 3. Data Flow Summary (one query, start to finish)

1. User submits a question via the Streamlit UI.
2. FastAPI receives it at `POST /query`.
3. Query is embedded (same model as ingestion) and run against ChromaDB for top-20 dense matches; in parallel, BM25 is run for top-20 sparse matches.
4. Results are fused via Reciprocal Rank Fusion into a single ranked candidate list.
5. Top-20 candidates are reranked by the cross-encoder; top 3–5 survive.
6. Surviving chunks + their source metadata (URL, section heading) are assembled into a prompt with a strict system instruction: *answer only from the provided context; if the context doesn't contain the answer, say so explicitly.*
7. The prompt is sent to the active LLM backend (Ollama locally, Groq in production).
8. The response, together with the source chunks used, latency, and token count, is returned to the frontend and logged.

---

## 4. Technology Stack Summary Table

| Layer | Choice | Cost | Key alternative considered & why rejected |
|---|---|---|---|
| Scraping | `requests` + `BeautifulSoup` | Free | Scrapy — overkill for a bounded, known doc set |
| Chunking | Custom structure-aware splitter (heading-based + overlap) | Free | LangChain's default `RecursiveCharacterTextSplitter` alone — used as a fallback for unstructured sections, but not the primary strategy, since hand-building the structure-aware layer is itself a resume-worthy engineering decision |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Free, local | OpenAI `text-embedding-3-small` — paid; Ollama embeddings — adds an unnecessary dependency for dev-only embedding |
| Vector store | ChromaDB (persisted local) | Free | Pinecone/Weaviate/Qdrant Cloud — free tiers exist but add external account dependency for no real benefit at this scale |
| Sparse retrieval | `rank_bm25` | Free | Elasticsearch — far too heavy for this scale |
| Reranker | BGE-reranker-v2 (open, local) | Free | Cohere Rerank 3.5 — paid API, best-in-class but violates $0 constraint |
| Orchestration | LangChain (used selectively, not as a black box) | Free | LlamaIndex — comparable; LangChain chosen for broader resume recognition and ecosystem docs |
| LLM (dev) | Ollama — Llama 3.1 8B or Qwen2.5 7B | Free, local | — |
| LLM (prod) | Groq API — `llama-3.1-8b-instant` | Free tier | OpenAI/Anthropic APIs — paid; Hugging Face Inference free tier — slower, less reliable rate limits |
| Backend | FastAPI | Free | Flask — less suited to async I/O-bound workload |
| Frontend | Streamlit | Free | Full React app — deferred to stretch goal given time budget |
| Evaluation | RAGAS (Groq as judge) | Free | DeepEval — comparable alternative, RAGAS chosen for broader name recognition on resumes |
| Containerization | Docker + docker-compose | Free | — |
| Hosting | Hugging Face Spaces (Docker SDK) | Free | Render/Railway/Fly.io free tiers — cannot run Ollama and are less recognized for AI-specific portfolio hosting |
| CI | GitHub Actions | Free (public repo) | — |

---

## 5. Key Architectural Risks (tracked and resolved in `memory.md` as they're addressed)

1. **Reranker compute on free CPU hosting.** BGE-reranker-v2 on HF Spaces' free CPU tier may add noticeable latency. Mitigation: benchmark it in Phase 4; if too slow, fall back to a smaller/faster reranker or skip reranking in the deployed version and document why (still a valid, measured engineering trade-off).
2. **Groq free-tier rate limits under demo load** (30 requests/min, ~14,400/day on `llama-3.1-8b-instant`). Acceptable for a portfolio demo; documented as a known production constraint, with a stated path to fix (Groq Developer tier) if this were a real product.
3. **Ollama/Groq output differences.** The two backends may answer slightly differently for the same prompt. Mitigation: final evaluation numbers reported in the README must be run against the *production* backend (Groq), not dev (Ollama), so the resume claim matches what a recruiter can actually go test live.
