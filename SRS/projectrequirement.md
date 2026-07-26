# Project Requirements Document (SRS)
## Production RAG Knowledge Assistant

**Version:** 1.0
**Status:** Draft — pending confirmation of target documentation source
**Owner:** [YOUR NAME]
**Last updated:** August 2026

---

## 1. Purpose

Build and publicly deploy a Retrieval-Augmented Generation (RAG) system that answers natural-language questions about a specific company's public API documentation, grounding every answer in retrieved source text rather than model memory, and measuring answer quality with a real evaluation harness (RAGAS).

This document is the single source of truth for *what* is being built. `architecture.md` and `design.md` describe *how*. `phases.md` describes *when*. `rules.md` describes *constraints on how we work*. `memory.md` tracks *decisions made and current state*.

---

## 2. Problem Statement

Developers integrating a third-party API frequently ask general-purpose LLMs (ChatGPT, Claude, etc.) for help. These models often **hallucinate**: they invent function names, cite deprecated API versions, or blend documentation from similar-but-different products. There is no verification step — the developer has no way to know if the answer came from real, current documentation or from the model's imperfect memory.

This project builds a narrow, grounded alternative: a system that can only answer using text it has actually retrieved from the real documentation, that shows its sources, and that has been measured (not assumed) to be faithful to those sources.

---

## 3. Target Corpus

**Confirmed:** Anthropic Claude API documentation (`docs.claude.com`) — final choice. Reasons: well-structured (clear heading hierarchy across API reference and guide pages, ideal for structure-aware chunking), no widely-publicized official RAG/doc-AI assistant currently competing with this project's exact pitch (unlike Stripe — see `literature-survey.md` §3 for why Stripe was ruled out), and scoped to a defined subset large enough to genuinely stress-test hybrid retrieval and reranking (target: API Reference — Messages, Models, Tool Use, Vision, Embeddings, Batches — plus core Guides — Prompt Engineering, Extended Thinking, Structured Outputs, Tool Use).

**Alternatives considered:** Stripe docs (ruled out — official Docs AI + published RAG tutorial already exists, see `literature-survey.md` §3), Twilio docs (viable backup, similarly well-structured, no known official competitor).

**Scoping note:** Anthropic's docs are a smaller corpus than Stripe's or Twilio's. This is a deliberate trade-off, not an oversight — the target page count (roughly 50–100 pages across the sections above) is intentionally chosen to be large enough that hybrid retrieval and reranking are genuinely necessary, while staying inside the 6-week time budget. If the confirmed subset comes in noticeably under ~50 pages once scraped, widen scope to additional guide sections (e.g., Agent SDK, MCP, Files API) before starting Phase 2, so retrieval quality has enough surface area to be meaningfully evaluated.

**Action required before Phase 1 starts:** Finalize the exact list of `docs.claude.com` URLs to scope (see updated open question in `memory.md`).

**Constraint:** We will only ingest publicly accessible documentation pages, respect `robots.txt`, and cache pages locally rather than re-scraping on every run. No authentication-gated content, no ToS-violating scraping. This is a non-negotiable rule (see `rules.md` §2).

---

## 4. In Scope

- Ingestion pipeline: fetch, clean, and structure-aware-chunk documentation pages
- Hybrid retrieval: dense vector search (embeddings) + sparse keyword search (BM25), fused
- Reranking of retrieved candidates before they reach the LLM
- Grounded answer generation with inline source citation (which doc section the answer came from)
- A local development inference path (Ollama) and a separate production inference path (Groq free API) — explicitly designed as two swappable backends, not one hardcoded choice
- REST API backend (FastAPI)
- Simple web frontend (Streamlit) to ask questions and see answers + sources
- Automated evaluation harness (RAGAS: faithfulness, answer relevancy, context precision/recall) with results tracked over time, not just a single run
- Unit + integration tests
- Dockerized, one-command local run (`docker-compose up`)
- CI pipeline (GitHub Actions) running tests on every push
- Public deployment (Hugging Face Spaces, Docker SDK)
- README with architecture diagram, setup instructions, and real before/after evaluation numbers

## 5. Out of Scope (explicitly, to prevent scope creep)

- Multi-tenant support / user accounts / auth — this is a single-purpose demo, not a SaaS product
- Support for arbitrary uploaded documents (corpus is fixed to the chosen company's docs, not user-uploaded files) — a "bring your own doc" feature is a stretch goal only, not a requirement
- Fine-tuning any model — we use off-the-shelf open models and embeddings only
- Multi-language support — English documentation only
- Streaming token-by-token UI responses — nice-to-have stretch, not required
- Payment, billing, or any handling of real user financial/personal data — this project must never require real credentials from anyone but the developer

---

## 6. Functional Requirements

| ID | Requirement | Priority |
|---|---|---|
| FR-1 | System can ingest a defined set of documentation URLs and produce cleaned, chunked, embedded text stored in a vector database | Must |
| FR-2 | Given a natural-language question, system retrieves relevant chunks using hybrid search (vector + BM25) | Must |
| FR-3 | Retrieved candidates are reranked before being passed to the LLM, and only the top-k (3–5) reach the prompt | Must |
| FR-4 | System generates an answer using only the retrieved context, with an explicit instruction to say "I don't know" rather than guess if the context is insufficient | Must |
| FR-5 | Every answer displays which source chunk(s)/doc section(s) it was grounded in | Must |
| FR-6 | System can be re-run in evaluation mode against a held-out question set and produce RAGAS scores (faithfulness, answer relevancy, context precision) | Must |
| FR-7 | LLM backend is swappable via config between local (Ollama) and hosted-free (Groq) without code changes | Must |
| FR-8 | System logs, per query: latency, retrieved chunk IDs, token usage, and the final RAGAS-relevant fields for later batch evaluation | Should |
| FR-9 | Frontend allows a user to type a question and see: the answer, the source snippets, and response latency | Must |
| FR-10 | System handles the "no relevant context found" case gracefully rather than hallucinating an answer | Must |
| FR-11 | (Stretch) User can filter/scope questions to a specific doc section (e.g., "Payments only") | Could |
| FR-12 | (Stretch) System supports basic conversational follow-up (multi-turn) using chat history | Could |

## 7. Non-Functional Requirements

| ID | Requirement | Target |
|---|---|---|
| NFR-1 | Deployed answer latency (p95) | Under 6 seconds end-to-end on the Groq backend |
| NFR-2 | Faithfulness (RAGAS) on held-out eval set | ≥ 0.85 after iteration (baseline will be measured and reported honestly even if lower) |
| NFR-3 | System must run fully offline for development (Ollama) with zero API cost | No paid API required for local dev loop |
| NFR-4 | System must run in a public deployment at zero recurring cost | Hugging Face Spaces free tier + Groq free tier |
| NFR-5 | Codebase must have automated tests covering chunking, retrieval, and the "no context found" path | ≥ 60% coverage on core `src/` logic, enforced in CI |
| NFR-6 | No secrets committed to version control | Enforced via `.env` + `.gitignore` + a pre-commit secret scan |
| NFR-7 | One-command reproducibility | `docker-compose up` brings up the full stack from a clean clone |
| NFR-8 | README must let a stranger (recruiter or interviewer) understand the architecture and results in under 3 minutes of reading | Peer-reviewed before final submission |

---

## 8. Success Criteria (how we know this project is "done" and resume-ready)

1. A public URL exists where anyone can ask a real question about the target company's API and get a grounded, cited answer.
2. RAGAS evaluation has been run at least twice (baseline vs. after retrieval-quality iteration) with numbers documented in the README — the resume bullet must be backed by a real, reproducible number, not an estimate.
3. The system correctly refuses to answer (rather than hallucinating) when asked something outside the corpus — this must be demonstrated with at least one logged example in the README.
4. The GitHub repo has: clean commit history, passing CI badge, architecture diagram, and a README a non-technical recruiter could skim in under 3 minutes and a technical interviewer could dig into for 20.
5. You can explain, unprompted, in under 90 seconds: the problem, the architecture, one hard technical decision (e.g., dev/prod inference split, or a chunking-strategy trade-off), and the measured result.

---

## 9. Constraints & Assumptions

- **Budget:** $0. Every component must have a genuinely free tier sufficient for a portfolio-scale project (not a "free trial that expires").
- **Time:** ~60–80 hours total, over 6 weeks, ~10 hrs/week. This bounds scope aggressively — see `rules.md` §1 (scope discipline) and `phases.md` for how this is allocated.
- **Solo developer**, Python-comfortable, new to: LangChain/LlamaIndex internals, vector databases, RAGAS, Docker-for-deployment, Ollama, Groq. These are the genuine learning-curve items to budget time for.
- **Assumption:** target documentation is static enough (doesn't change hour-to-hour) that a periodic re-ingestion job, not real-time sync, is sufficient.

---

## 10. Open Questions (tracked live in `memory.md`)

- [ ] Final confirmation of target company docs and exact URL scope
- [ ] Exact held-out evaluation question set — how many questions, how are they authored (see `design.md` §6)
- [ ] Whether HF Spaces free CPU tier is sufficient for the reranker model, or whether reranking needs to be skipped/lightened in the deployed version (flagged as a technical risk in `memory.md`)
