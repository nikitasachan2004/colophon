# Memory Document
## Colophon — Production RAG Knowledge Assistant — Living Project State

**Purpose:** This file is the persistent memory across sessions (yours and any AI assistant's). Update it every time a real decision is made, a phase is completed, or a risk materializes. Do not rely on conversation history to remember this — write it down here, immediately, when it happens.

---

## Current Status

**Phase:** Phase 4 COMPLETE — Phase 5 (Deployment, CI) in progress
**Last updated:** September 4, 2026
**Overall project start date:** August 30, 2026
**Target completion:** ~October 11, 2026 (6 weeks from start)

---

## Open Questions (resolved)

- [x] **Target company docs — CONFIRMED: Anthropic Claude API documentation (`platform.claude.com/docs`).** Stripe was considered first but ruled out (official Docs AI + published RAG tutorial already exists). A first implementation pass was built against Stripe by mistake and the corpus was swapped. Canonical domain is `platform.claude.com/docs` — `docs.claude.com` is a 301 redirect to the same content; `source_url` metadata and `eval_questions.json` both use the canonical domain.
- [x] **URL list confirmed:** 77 URLs across 9 categories; 2,621 chunks indexed. Above the 50-page minimum threshold.
- [x] **Eval question set size confirmed:** 25 held-out questions (20 in-domain + 5 OOD). Covers all major topic categories.

---

## Decisions Log

| Date | Decision | Reasoning | Reference |
|---|---|---|---|
| 2026-08-30 | Chose RAG Knowledge Assistant as the flagship project over the other 5 candidates | Best match to 2026 AI hiring signals (RAG/vector DB most-requested keyword), fastest path to a genuinely impressive deployed demo within the time budget | See original project-selection research |
| 2026-08-30 | Fully free/local-first stack: Ollama (dev) + Groq (prod), Chroma, sentence-transformers, BGE-reranker-v2, HF Spaces | Zero budget constraint; each choice verified against 2026 free-tier terms at research time | `architecture.md` §2, §4 |
| 2026-08-30 | Structure-aware chunking (heading-based + code-block-atomic + 10-15% overlap) as primary strategy, not fixed-size | 2026 benchmarks show chunking strategy is the single highest-leverage lever in RAG quality | `design.md` §3 |
| 2026-08-30 | Hybrid retrieval (dense + BM25 via RRF) + cross-encoder reranking, retrieve-20-rerank-to-5 | Standard 2026 production recipe; catches exact-match API method names that vector-only misses | `architecture.md` §2.3, §2.4 |
| 2026-08-30 | RAGAS chosen over DeepEval for evaluation | Broader resume/name recognition; comparable capability | `architecture.md` §2.9 |
| 2026-08-31 | First implementation pass (Phase 0-3) built against `docs.stripe.com`, ~1,751 chunks indexed | Built before the Stripe-vs-Anthropic decision was finalized | Full stack works end-to-end; only corpus swap required |
| 2026-08-31 | **Corpus swapped: Stripe → Anthropic Claude API docs** | Avoids "official product already exists" comparison that ruled out Stripe | `projectrequirement.md` §3 |
| 2026-08-31 | **Project renamed: GroundTruth → Colophon** | A colophon is the note at the end of an old manuscript stating its true origin — exactly what this system does for every answer. Stronger brand narrative. | All frontend, API, docker, CI files updated |
| 2026-08-31 | Canonical domain fixed: `docs.claude.com` → `platform.claude.com/docs` | `docs.claude.com` is a 301 redirect; all chunk metadata and eval URLs corrected to canonical. Metadata-only fix — content was always from the live platform.claude.com site. | All 2,621 ChromaDB chunks updated in-place; BM25 rebuilt |
| 2026-08-31 | RAGAS judge model fixed as `llama3.1:8b` via Ollama (local) for all four ablation stages | Early runs used various Groq models (allam-2-7b, gpt-oss-20b, qwen3.8-27b) and hit per-model 200k TPD limits mid-run, making stages non-comparable. Local Ollama: zero quota, no TPD, pinned across all four stages. | `run_eval.py` `compute_ragas_scores()` |
| 2026-08-31 | RRF confidence-threshold bug fixed | `_retrieve_hybrid_no_rerank` was applying CONFIDENCE_THRESHOLD=0.35 to RRF scores (~0.016 max) — zeroing out all hybrid results. Fixed to use dense-score as the context-found signal. | `src/evaluation/run_eval.py` |
| 2026-09-02–04 | **Phase 4 clean ablation redo** | All four stages re-run with connection-error retry, Ollama judge pinned. Three rounds of connection errors and TPD exhaustion cleared before clean runs produced. | See debugging narrative below |
| 2026-09-04 | **Final production config locked: `RERANK_CANDIDATE_POOL=10`, CPU reranker for deployment** | `RERANK_CANDIDATE_POOL=10` already in `.env` default. CPU reranker (`FORCE_CPU_RERANKER=1`) is required for HF Spaces free-tier deployment (no MPS/CUDA). Live endpoint benchmark confirms p50=7,618ms, p95=11,013ms — this is the honest latency for deployed config. NFR-1 miss documented as a free-hardware constraint. | `latency_optionB_benchmark.log` |
| 2026-09-06 | **Production Reranker Swapped to `cross-encoder/ms-marco-MiniLM-L-6-v2` (Render 512MB RAM Fix)** | Production deployment (Render free tier, 512MB RAM) uses `cross-encoder/ms-marco-MiniLM-L-6-v2` (22M params) as the reranker instead of `BAAI/bge-reranker-v2-m3` (560M params) used during Phase 4's RAGAS evaluation, due to a hard memory ceiling on free-tier hosting. The RAGAS quality scores reported (Faithfulness=0.706 etc.) reflect the larger BGE reranker and have not been re-validated against the lightweight production reranker. A quick recall@5/OOD sanity check confirmed the lightweight reranker is not badly broken (Recall@5=0.9000, OOD Accuracy=0.8000), but a full RAGAS comparison between the two rerankers was not performed. | `src/config.py`, `src/retrieval/reranker.py` |

---


## Technical Risks — Live Tracker

| Risk | Status | Notes |
|---|---|---|
| Reranker latency on free CPU hosting (HF Spaces) | **CONFIRMED MISS of NFR-1** | Live `/query` endpoint benchmark (CPU reranker, pool=10): p50=7,618ms, p95=11,013ms. NFR-1 target is p95 < 6,000ms. Gap is ~5s at p95. Root cause: BGE-reranker-v2-m3 on CPU is ~6-7s per query after warm-up. This is a free-hardware constraint — not a code bug. Mitigations considered: (1) swap to `cross-encoder/ms-marco-MiniLM-L-6-v2` (smaller, faster), (2) disable reranker for deployed demo and accept lower quality, (3) accept the miss and disclose honestly in README. Decision: accept and disclose — quality is more valuable than meeting an aggressive latency target on free hardware. |
| Groq free-tier rate limits (TPD) | **CONFIRMED CONSTRAINT — causes eval failures** | 200k tokens/day per model, refills at 2.31 tokens/sec. Heavy eval sessions (25 questions × ~2,500 tokens each = ~50k tokens) exhaust the window. OptionB re-run was blocked twice by TPD exhaustion. Production demo at portfolio scale (single-digit queries/day) is fine. Not suitable for any real traffic. |
| Groq free-tier rate limits (TPM) | MITIGATED | 8,000 tokens/minute per model. Fixed by 70s inter-question sleep in eval harness (refills ~162 tokens in 70s — net deficit, but the 70s gap lets earlier TPM windows clear). |
| RAGAS 0.3.x + Ollama compatibility | MITIGATED | `max_workers=1` required (concurrent calls cause systematic TimeoutErrors on M4 — second worker starves). `timeout=300s`. `MAX_CTX_CHARS=300` prevents `LLMDidNotFinishException`. NaN rows from timeouts handled by `nanmean` aggregation. |
| Eval question 006 and 018 persistent misses | KNOWN LIMITATION | eval-006 (Claude 3.7 Sonnet max output tokens) and eval-018 (anthropic-version header value) miss across all retrieval configs. Content exists in corpus but is buried in large chunks that score below top-5. |

---

## Deferred Ideas (explicitly out of current scope)

- Multi-turn conversational follow-up (FR-12, stretch)
- Doc-category filtering in the UI (FR-11, stretch)
- "Bring your own document" upload feature
- Full React frontend (Streamlit is the committed choice for the 6-week budget)
- Streaming token-by-token responses

---

## Glossary

- **RAG** — Retrieval-Augmented Generation: retrieve relevant text first, then have the LLM answer using only that retrieved text.
- **Hybrid search** — combining dense vector similarity search with sparse keyword search (BM25), fused via Reciprocal Rank Fusion.
- **Reranking** — a second, more expensive scoring pass (cross-encoder) applied to a small candidate set to improve final ordering before it reaches the LLM.
- **Faithfulness (RAGAS metric)** — measures whether the generated answer's claims are actually supported by the retrieved context (the core anti-hallucination check).
- **Confidence threshold** — the score cutoff below which the system declares "no relevant context found" instead of attempting an answer (`design.md` §4).

---

## Phase Completion Log

| Phase | Completed | Hours (actual vs. budget) | Notable deviations |
|---|---|---|---|
| Phase 0 | 2026-08-30 | ~2 hrs / 2 budgeted | Built against Stripe by mistake; corpus swap added |
| Phase 1 | 2026-08-31 | ~10 hrs / 10 budgeted | 77 URLs, 2,621 chunks, all metadata correct |
| Phase 2 | 2026-08-31 | ~10 hrs / 10 budgeted | Hybrid retrieval + RRF + BGE reranker working |
| Phase 3 | 2026-08-31 | ~10 hrs / 10 budgeted | FastAPI + Streamlit + Groq + OOD refusal path |
| Phase 4 | 2026-09-04 | ~20 hrs / 12 budgeted | 3 rounds of reruns to get clean 0-error results; +8hrs debugging |
| Phase 5 | 2026-09-05 | — | Next.js 16 frontend (frontend-web/). Streamlit archived to frontend/legacy_streamlit/. First git commit: 08be0e4. HF Space + Vercel deployment pending. |
| Phase 6 | — | — | — |

---

## Phase 4 Final Ablation Results (CANONICAL — September 2–4, 2026)

**Setup:** 25 held-out questions (20 in-domain, 5 OOD). Generation: Groq `openai/gpt-oss-120b`. RAGAS judge: `llama3.1:8b` via Ollama (local, zero quota, pinned across all four stages). All four stages verified at 0 generation errors before RAGAS scoring.

| Stage | Chunking | Retrieval | Reranking | Recall@5 | OOD Acc | Faithfulness | Ans Relevancy | Ctx Precision | Ctx Recall | Avg Latency | Result File |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | Naive fixed-size (512 tok, 10% overlap) | Vector-only | None | 0.800 | 1.000 | 0.6079 | 0.6772 | 0.7483 | 0.9969 | 1,144 ms | `phase4_baseline_naive_chunks,_vector-only_20260902_204914.json` |
| + Structure-Aware Chunking | Heading-split, code-atomic, 12% overlap | Vector-only | None | 0.800 | 1.000 | 0.6853 | 0.6513 | 0.7483 | 0.9969 | 1,475 ms | `phase4_plus_structure-aware_chunking_20260903_140340.json` |
| + Hybrid Retrieval | Structure-aware | Dense + BM25 + RRF | None | 0.900 | 1.000 | 0.5638 | 0.7479 | 0.7994 | 0.9500 | 1,602 ms | `phase4_plus_hybrid_retrieval_vector_plus_bm25_plus_rrf_20260903_180606.json` |
| **+ Reranking (Final)** | Structure-aware | Dense + BM25 + RRF | BGE-reranker-v2-m3 | **0.900** | **1.000** | **0.7059** | **0.7455** | **0.8609** | **0.9367** | 33,308 ms† | `phase4_plus_reranking_final___production_20260903_235247.json` |

†Eval harness latency includes MPS reranker warm-up and does not reflect deployed endpoint latency. See live benchmark below.

**Live endpoint latency (production config — CPU reranker, pool=10):**
- Benchmark: `src/evaluation/results/latency_optionB_benchmark.log` (20 in-domain questions, `/query` endpoint, Sep 4 2026)
- p50 = 7,618 ms
- p95 = 11,013 ms
- NFR-1 target p95 < 6,000 ms: **FAIL by ~5s**
- This is the deployed latency on free CPU. Not a code bug — BGE-reranker-v2-m3 takes ~6-7s on CPU after warm-up. Disclosed honestly in README.

**RAGAS NaN rows:** Stage 4 had 1 NaN faithfulness row (eval-006: known persistent miss); Stage 1 had 2 NaN rows. All handled by `nanmean` aggregation — logged, not silently dropped.

---

## Phase 4 Debugging Narrative

This is real engineering signal — kept as a record of what actually happened.

**Round 1 (Aug 31):** Initial runs used `allam-2-7b` via Groq as RAGAS judge. Hit 200k TPD mid-run on multiple Groq models, making stages 2-4 RAGAS scores non-comparable (each stage used a different judge as quotas exhausted). Also had a confidence-threshold bug in `_retrieve_hybrid_no_rerank` applying `CONFIDENCE_THRESHOLD=0.35` to RRF scores (max ~0.016), zeroing all hybrid results. Both bugs produced the Aug 31 numbers in the old result table.

**Round 2 (Sep 2):** Switched to Ollama `llama3.1:8b` as judge (zero quota). Venv `.so`/`.dylib` files had `com.apple.quarantine` xattrs blocking load — stripped with `xattr -d`. Python 3.12 symlink broken (old framework removed) — repointed to Homebrew 3.12. Ran Stage 1 clean (25/25, 0 errors). Stages 2-4 hit Groq connection errors (transient network drops) because no retry logic existed. Files named `_INVALID_connection_errors` and discarded.

**Round 3 (Sep 2–3):** Added retry logic (3 retries, 30s default, parses `Please try again in Xm Ys` from 429 messages). Added 70s inter-question sleep to stay under 8k TPM. Fixed connection error catch alongside 429 catch. Stages 2-4 ran clean overnight.

**Round 4 (Sep 4 — OptionB attempts):** Attempted to validate CPU+pool=10 config (HF Spaces deployment equivalent) with the same `gpt-oss-120b` gen model. Two attempts, both failed. The RAGAS scores from the corrupted run (`phase4_plus_reranking_optionB_pool10_20260904_065957.json`) show Faithfulness=0.247, Ans Relevancy=0.227 — these numbers are meaningless; they reflect error strings (`[Generation error: 429-TPD]`) being fed into RAGAS, not real answers. **There are no valid Option B RAGAS scores. None were produced.**

- *First attempt:* launched 4.5h after the overnight Stage 4 run that had already consumed 199,464/200,000 TPD tokens on `gpt-oss-120b`. Only 536 TPD tokens remained — enough for 5 questions, then all subsequent LLM calls hit 429-TPD immediately. 14/25 generation errors. The corrupted file was retained with its generation-error count explicitly recorded to avoid mistaking it for a valid result.
- *Second attempt:* TPD appeared to reset (small test calls succeeded), but real eval-size prompts (~2,500 tokens each) immediately hit the wall again. eval-001 completed (16,681ms, clean); eval-002 hit a 1,409-second 429-sleep. Stopped. TPD refill rate is 2.31 tokens/sec — each question uses ~2,500 tokens but the 70s inter-question pacing only refills ~162 tokens. Net deficit of ~2,338 tokens per question means the run would need ~18-minute gaps between questions: 20 questions × 18 min = 6+ hours. Not viable.

**Consequence:** The Option B RAGAS comparison (CPU reranker vs MPS reranker quality) was never completed and is honestly reported as missing — not approximated from the canonical Stage 4, not fabricated. The canonical Stage 4 RAGAS scores (faith=0.706, ansrel=0.746, ctx_prec=0.861, ctx_rec=0.937) remain the only valid RAGAS measurement and come from the MPS-reranker run, which uses the same retrieval config (RERANK_CANDIDATE_POOL=10) but different hardware.

**Ollama RAGAS concurrency:** Running RAGAS with `max_workers=2` on M4 caused every even-numbered job to time out at 120s (second concurrent call starved). Fixed with `max_workers=1, timeout=300`.

---

## Task 2 Decision: Final Production Config

**Chosen: `RERANK_CANDIDATE_POOL=10`, `FORCE_CPU_RERANKER=1` (deployment), `gpt-oss-120b` (generation)**

**Why CPU reranker, not MPS:** HF Spaces CPU Basic has no GPU, no MPS, no CUDA — the BGE reranker runs on CPU there regardless of what the code requests. Reporting MPS latency numbers (~33s per query on an M4) as the "production latency" would be dishonest — no user of the deployed app ever sees 33s. The latency benchmark used `FORCE_CPU_RERANKER=1` specifically because it matches the deployment hardware. The p50=7,618ms, p95=11,013ms numbers are what users actually experience on the deployed HF Space, not what a local M4 Mac measures. Using CPU for benchmarking isn't pessimistic conservatism — it's the only number that's true for the deployed system.

This is an engineering call: benchmark the hardware your users get, not the hardware you have. The MPS measurement (33s avg in the eval harness) is valid for local dev context but irrelevant to the deployment story.

- `RERANK_CANDIDATE_POOL=10` was already the `.env` default. No change needed.
- No further pool-size tuning. The latency miss (p95=11s vs p95<6s NFR-1 target) is a free-CPU hardware constraint. Chasing pool=5 or pool=7 would risk degrading retrieval quality to solve a constraint that can't be fully resolved at zero cost on free hardware.
- This is the final decision. No more ablation loops.

---

## Resume Bullet (verified, backed by result files)

> Built a production RAG knowledge assistant over Anthropic Claude API docs (2,621 chunks, 77 pages); hybrid retrieval (dense + BM25 + RRF) + BGE cross-encoder reranking achieved **Recall@5 = 0.90**, **Faithfulness = 0.706**, **Context Precision = 0.861**, **Answer Relevancy = 0.746** on a 25-question held-out RAGAS evaluation (Ollama llama3.1:8b judge, 0 generation errors, all four ablation stages). Deployed latency on free CPU hosting: p50=7,618ms, p95=11,013ms.

**Caveats to state if asked:**
- RAGAS scores are from the MPS-reranker run (local M4), not the CPU-reranker deployment config — those runs failed due to Groq TPD exhaustion and were not completed.
- The p50/p95 latency is from the CPU-reranker live endpoint benchmark, which is the honest deployment measurement.
- Faithfulness=0.706 with Ollama judge. An earlier run (Aug 31, Groq allam-2-7b judge) showed 1.000 — that number is superseded; the judge and RAGAS settings were different and the scores aren't comparable.
