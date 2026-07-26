# Phases Document
## Production RAG Knowledge Assistant — 6-Week Execution Plan

Budget: ~10 hrs/week, ~60–70 hrs total (holding 10–15 hrs in reserve across phases for the inevitable overrun — see `rules.md` §1 on scope discipline). Each phase has a **Definition of Done** — do not move to the next phase until it's met, even if that means cutting a stretch item.

---

## Phase 0 — Setup & Confirmation (before Week 1, ~2 hrs)

- [ ] Confirm target company docs + exact URL subset (see `memory.md` open question)
- [ ] Create GitHub repo, set up folder structure per `design.md` §1
- [ ] Install Ollama locally, pull `llama3.1:8b`, verify it runs
- [ ] Create Groq account, get free API key, verify a test call works
- [ ] Set up `.env`, `.gitignore`, `requirements.txt` skeleton
- **Definition of Done:** empty project scaffold exists, both LLM backends are reachable from a throwaway test script, repo is initialized with a first commit.

---

## Phase 1 — Ingestion Pipeline (Week 1, ~10 hrs)

- [ ] Write scraper for the confirmed doc subset (respect robots.txt, cache raw HTML to `data/raw/`)
- [ ] Write HTML → clean markdown cleaner (strip nav/footer/ads, preserve headings and code blocks)
- [ ] Implement structure-aware chunker per `design.md` §3
- [ ] Implement embedder (sentence-transformers) + write to ChromaDB
- [ ] Unit tests: `test_chunker.py` (code blocks never split, overlap present, heading metadata attached)
- **Definition of Done:** Running the ingestion script end-to-end on the real doc subset produces a populated, persisted ChromaDB with correct metadata on every chunk — verified by manually inspecting 5 random chunks.
- **Risk to watch:** doc site structure might not be as clean as expected — budget slack here first if something in Phase 1 runs long, since everything downstream depends on chunk quality.

---

## Phase 2 — Retrieval Core (Week 2, ~10 hrs)

- [ ] Implement dense vector search against Chroma
- [ ] Implement BM25 sparse search
- [ ] Implement Reciprocal Rank Fusion
- [ ] Implement reranker (BGE-reranker-v2) — first get it running locally, benchmark its latency
- [ ] Unit tests: `test_fusion.py`, `test_reranker.py`
- [ ] Manual sanity check: run 10 known questions through retrieval only (no generation yet) and eyeball whether the right chunks come back in the top 5
- **Definition of Done:** for at least 8/10 manually-checked questions, the correct source chunk appears in the top-5 post-rerank results.

---

## Phase 3 — Generation & API (Week 3, ~10 hrs)

- [ ] Implement `llm_client.py` backend-agnostic interface + `ollama_backend.py` + `groq_backend.py`
- [ ] Implement prompt templates per `design.md` §5
- [ ] Implement the "no context found" confidence-threshold path (FR-10)
- [ ] Build FastAPI `/query` and `/health` endpoints per `design.md` §9
- [ ] Integration tests: `test_query_pipeline.py` (end-to-end happy path), `test_no_context_found.py` (deliberately out-of-corpus question returns the "I don't know" response, not a hallucination)
- **Definition of Done:** `curl -X POST /query` against a running local server returns a correct, cited answer for a known question, and the correct refusal for an out-of-corpus question.

---

## Phase 4 — Evaluation & Iteration (Week 4, ~12 hrs — this is the phase that actually produces your resume metric, protect this time)

- [ ] Write the held-out evaluation question set (25–40 questions, by hand, per `design.md` §2.4)
- [ ] Implement `run_eval.py` with RAGAS scoring
- [ ] Run **baseline** eval (naive fixed-size chunking, vector-only, no reranking) — record honestly, even if the number is mediocre
- [ ] Re-run eval after switching in structure-aware chunking — record delta
- [ ] Re-run eval after adding hybrid retrieval — record delta
- [ ] Re-run eval after adding reranking — record delta
- [ ] Tune `CONFIDENCE_THRESHOLD`, `CHUNK_SIZE_TOKENS`, `CHUNK_OVERLAP_PCT` against the eval set; log every experiment (values + resulting scores) in `memory.md`
- **Definition of Done:** you have a table of at least 4 eval runs (baseline → final) with real numbers, and you can explain in one sentence why each change moved the score.
- **This is the single most important phase for your resume bullet — do not skip or rush it even if earlier phases ran long.**

---

## Phase 5 — Frontend, Deployment, CI (Week 5, ~10 hrs)

- [ ] Build Streamlit frontend per `design.md` §8
- [ ] Write Dockerfiles for API + frontend, write `docker-compose.yml`
- [ ] Verify `docker-compose up` works end-to-end from a clean clone
- [ ] Set `LLM_BACKEND=groq` for the deployed config, verify it works
- [ ] Push to Hugging Face Spaces (Docker SDK), verify the public URL works
- [ ] Set up GitHub Actions CI running `pytest` on every push
- **Definition of Done:** a stranger with no context can open the public HF Spaces URL, ask a real question, and get a correct, cited, fast answer — test this yourself from a phone on mobile data as a final check.

---

## Phase 6 — Documentation & Polish (Week 6, ~8 hrs)

- [ ] Draw the architecture diagram (can reuse/clean up the one in `architecture.md`)
- [ ] Write final README: problem, architecture diagram, setup instructions, the eval-run table from Phase 4, one documented example of the "I don't know" refusal working, known limitations
- [ ] Record a 60–90 second demo video/GIF
- [ ] Do a full re-read of the repo pretending to be an interviewer: does every claim in the README have receipts (a test, a logged eval run, a screenshot)?
- [ ] Write and rehearse your 60–90 second spoken explanation (problem → architecture → one hard decision → measured result) — this is explicitly part of the project per `projectrequirement.md` §8
- **Definition of Done:** everything in `projectrequirement.md` §8 (Success Criteria) is checked off.

---

## Contingency Rules

- If you're behind by end of Week 3: cut FR-11/FR-12 (stretch filters, multi-turn) entirely — they were never in scope (see `projectrequirement.md` §5).
- If Phase 4 reveals the reranker is too slow on free CPU hosting: document the trade-off and ship without it in production (still measure and report the eval score *with* it locally, so the engineering work isn't wasted — just note the deployed version differs, which is itself an honest, interview-worthy trade-off).
- Never sacrifice Phase 4 (evaluation) time to finish Phase 5/6 faster — a project with no real numbers is a demo, not a resume-worthy engineering project, per `projectrequirement.md` §2.
