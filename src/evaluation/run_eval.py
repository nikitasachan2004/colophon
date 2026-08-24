"""
Evaluation harness — Phase 4 ablation study for Colophon RAG pipeline.

Runs the held-out question set through four experimental configurations,
computing RAGAS metrics (faithfulness, answer_relevancy, context_precision,
context_recall) plus recall@k and OOD refusal accuracy for each.

Why these four runs (phases.md §4):
  Baseline   — fixed-size chunking (512 tokens, no heading awareness), vector-only, no reranker
  Run 2      — structure-aware chunking (current corpus), vector-only, no reranker
  Run 3      — structure-aware chunking, hybrid (vector + BM25 + RRF), no reranker
  Run 4/Final— structure-aware chunking, hybrid, + cross-encoder reranking  ← production config

Rules (rules.md §5): every metric in README/resume must come from a logged run here.
Never fabricate or estimate — run real experiments, write real numbers.
"""
from __future__ import annotations

import json
import logging
import os
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Suppress noisy deprecation warnings from langchain ecosystem
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*langchain_community.*")

from dotenv import load_dotenv

load_dotenv()

# ── Offline HuggingFace mode ──────────────────────────────────────────────────
# The embedding model (all-MiniLM-L6-v2) is already fully cached locally.
# Without these flags, HuggingFace Hub tries to reach huggingface.co on every
# load, hangs for 90s retrying with no network, then falls back to cache anyway.
# Setting these at import time makes the load instant and deterministic.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from src.config import PROJECT_ROOT, CONFIDENCE_THRESHOLD, RERANK_FINAL_K
from src.retrieval.bm25_search import bm25_search
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.reranker import rerank
from src.retrieval.vector_search import vector_search, SearchResult
from src.generation import llm_client
from src.generation.prompts import NO_CONTEXT_RESPONSE, SYSTEM_PROMPT, build_user_prompt

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("evaluation")

EVAL_FILE = PROJECT_ROOT / "src" / "evaluation" / "eval_questions.json"
RESULTS_DIR = PROJECT_ROOT / "src" / "evaluation" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Collection override context manager ──────────────────────────────────────
# Used to route Stage 1 (Baseline) to the 'baseline_naive_chunks' collection
# instead of the production 'claude_docs' collection.  vector_search() always
# calls src.ingestion.embedder.get_collection() — we patch it here for the
# duration of a single run.

import contextlib
import importlib


@contextlib.contextmanager
def _use_collection(collection_name: str):
    """
    Temporarily override the ChromaDB collection used by vector_search and bm25_search.

    On entry: monkey-patches src.ingestion.embedder.get_collection to return the
              named collection, and clears the BM25 module-level cache so it
              rebuilds from the right collection.
    On exit:  restores the originals.
    """
    import src.ingestion.embedder as embedder_mod
    import src.retrieval.bm25_search as bm25_mod

    original_get_collection = embedder_mod.get_collection

    def _patched_get_collection():
        import chromadb
        from src.config import CHROMA_PERSIST_DIR
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        return client.get_collection(name=collection_name)

    try:
        embedder_mod.get_collection = _patched_get_collection
        # Reset BM25 cache so it reloads from the correct collection
        bm25_mod._bm25_index = None
        bm25_mod._bm25_corpus_ids = []
        bm25_mod._bm25_corpus_texts = []
        bm25_mod._bm25_corpus_metadata = []
        # Also clear the cached bm25_index.pkl (force rebuild from new collection)
        _saved_path = bm25_mod.BM25_INDEX_PATH
        bm25_mod.BM25_INDEX_PATH = bm25_mod.BM25_INDEX_PATH.parent / f"bm25_index_{collection_name}.pkl"
        yield
    finally:
        embedder_mod.get_collection = original_get_collection
        bm25_mod.BM25_INDEX_PATH = _saved_path
        # Reset BM25 cache again so next run rebuilds from production collection
        bm25_mod._bm25_index = None
        bm25_mod._bm25_corpus_ids = []
        bm25_mod._bm25_corpus_texts = []
        bm25_mod._bm25_corpus_metadata = []

EVAL_FILE = PROJECT_ROOT / "src" / "evaluation" / "eval_questions.json"
RESULTS_DIR = PROJECT_ROOT / "src" / "evaluation" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── RAGAS setup ─────────────────────────────────────────────────────────────

def _aggregate_ragas_df(result: Any) -> dict[str, float]:
    """
    Aggregate a RAGAS EvaluationResult into per-metric means.

    RAGAS 0.3.9 behaviour (verified in tests/unit/test_eval_aggregation.py):
    - result["metric"] returns a List[float] — NOT a scalar.
      float(result["metric"]) raises TypeError — do not use this path.
    - result.to_pandas() returns one row per question × one column per metric,
      plus the original dataset columns (question, answer, contexts, …).
    - np.nanmean(df[col].values) averages DOWN rows for a single column,
      correctly handling NaN cells (timeouts) per-metric rather than per-row.

    Args:
        result: A RAGAS EvaluationResult (or any mock with .to_pandas()).

    Returns:
        Dict of metric_name → float. Missing columns are absent (not 0.0).
        All-NaN columns return float("nan") — a visible NaN is honest.
    """
    import numpy as np

    METRIC_COLS = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]
    df = result.to_pandas()
    scores: dict[str, float] = {}
    for col in METRIC_COLS:
        if col not in df.columns:
            continue
        with warnings.catch_warnings():
            # nanmean of an all-NaN column emits RuntimeWarning: Mean of empty slice.
            # This is expected and handled — the result is nan, not a crash.
            warnings.simplefilter("ignore", RuntimeWarning)
            scores[col] = float(np.nanmean(df[col].values))
    return scores


def _build_ragas_llm():
    """
    Return a LangChain-wrapped LLM for RAGAS scoring.

    Uses RAGAS_JUDGE_BACKEND / RAGAS_JUDGE_MODEL from config — deliberately
    isolated from LLM_BACKEND / GROQ_MODEL (the production generation path).
    This ensures the same judge model is used across all four ablation stages
    and that Groq's production-model quota is not consumed by scoring calls.

    See config.py for the full architecture decision comment.
    """
    from src.config import (
        RAGAS_JUDGE_BACKEND,
        RAGAS_JUDGE_MODEL,
        RAGAS_JUDGE_OLLAMA_URL,
        GROQ_API_KEY,
    )

    if RAGAS_JUDGE_BACKEND == "ollama":
        try:
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model=RAGAS_JUDGE_MODEL,
                base_url=RAGAS_JUDGE_OLLAMA_URL,
                temperature=0,
            )
        except ImportError:
            logger.warning(
                "langchain_ollama not installed — falling back to groq judge"
            )

    # Groq judge path (default, or Ollama fallback)
    from langchain_groq import ChatGroq
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set — required for RAGAS judge scoring")
    return ChatGroq(
        model=RAGAS_JUDGE_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0,
        max_tokens=2048,  # Faithfulness enumerates statements — needs room; 512/1024 caused LLMDidNotFinishException
        max_retries=6,    # Retry 429s at the LangChain layer before RAGAS timeout fires
    )


def _build_ragas_embeddings():
    """Return LangChain-wrapped embeddings for RAGAS context scoring.

    HF_HUB_OFFLINE=1 is set at module top so this loads from local cache
    instantly — no network call, no 90s retry storm when internet is down.
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def compute_ragas_scores(
    eval_rows: list[dict[str, Any]],
) -> dict[str, float]:
    """
    Compute RAGAS faithfulness, answer_relevancy, context_precision, context_recall
    over in-domain (non-OOD) questions only.

    Judge model: RAGAS_JUDGE_MODEL (config.py) — pinned for all four ablation runs.
    Returns a dict of metric_name → float (0–1), or empty dict on failure.

    Per-question NaN handling: if a single metric call fails (timeout, malformed
    response), the cell is NaN in the RAGAS output. _aggregate_ragas_df uses
    np.nanmean so the aggregate stays valid and NaN cells are logged explicitly
    rather than silently omitting the entire metric.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from datasets import Dataset
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.run_config import RunConfig
    except ImportError as e:
        logger.warning("RAGAS not available: %s", e)
        return {}

    # Groq doesn't support n>1; answer_relevancy defaults to strictness=3 (n=3).
    # Force n=1 so all metrics work within Groq's constraints.
    answer_relevancy.strictness = 1

    # Filter to in-domain questions that got an answer (not refusals)
    scored_rows = [
        r for r in eval_rows
        if r["category"] != "out_of_domain"
        and r["context_found"]
        and r.get("contexts")
    ]

    if not scored_rows:
        logger.warning("No scoreable rows for RAGAS (all OOD or no context found)")
        return {}

    # Truncate contexts to 300 chars each. With 5 chunks that's ~1500 chars ≈ 375 tokens
    # of context per RAGAS call. Keeps each call well within 8k TPM budget
    # (~600-800 tokens total per call). Avoids LLMDidNotFinishException.
    MAX_CTX_CHARS = 300

    dataset = Dataset.from_dict({
        "question":     [r["question"] for r in scored_rows],
        "answer":       [r["answer"] for r in scored_rows],
        "contexts":     [[c[:MAX_CTX_CHARS] for c in r["contexts"]] for r in scored_rows],
        "ground_truth": [r["ground_truth"] for r in scored_rows],
    })

    from src.config import RAGAS_JUDGE_BACKEND, RAGAS_JUDGE_MODEL
    print(f"\n  [RAGAS judge] backend={RAGAS_JUDGE_BACKEND!r} model={RAGAS_JUDGE_MODEL!r}"
          f"  ← fixed for all ablation stages, not the production model")

    try:
        llm = LangchainLLMWrapper(langchain_llm=_build_ragas_llm())
        emb = LangchainEmbeddingsWrapper(embeddings=_build_ragas_embeddings())

        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=llm,
            embeddings=emb,
            raise_exceptions=False,
            show_progress=True,
            # max_workers=1: Ollama llama3.1:8b is single-threaded on local CPU/MPS.
            # Running 2 workers concurrently starves the second call and causes
            # systematic TimeoutErrors on every even-numbered job (observed Sep 2026).
            # timeout=300: 8B model needs ~2-3 min per faithfulness call on MPS.
            run_config=RunConfig(max_workers=1, timeout=300),
        )

        # Aggregate using to_pandas() + nanmean — the only correct path for RAGAS 0.3.9.
        # result["metric"] returns List[float]; float(list) raises TypeError.
        # See tests/unit/test_eval_aggregation.py for the full verification.
        scores = _aggregate_ragas_df(result)

        # Log any NaN cells (per-question failures) for visibility
        import numpy as np
        df = result.to_pandas()
        for col in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            if col in df.columns:
                nan_rows = df[df[col].isna()]
                if not nan_rows.empty:
                    logger.warning(
                        "RAGAS metric '%s': %d NaN row(s) — questions: %s",
                        col,
                        len(nan_rows),
                        nan_rows.get("question", nan_rows.index).tolist()[:5],
                    )

        logger.info("RAGAS scores: %s", scores)
        return scores
    except Exception as exc:
        logger.warning("RAGAS evaluation failed: %s", exc)
        return {}


# ── Retrieval modes ──────────────────────────────────────────────────────────

def _retrieve_vector_only(query: str, top_k: int) -> tuple[list[SearchResult], bool]:
    """Dense vector search only — no BM25, no fusion, no reranker."""
    hits = vector_search(query, top_k=top_k)
    # Simple threshold: if best score is below CONFIDENCE_THRESHOLD, no context
    if not hits or hits[0].score < CONFIDENCE_THRESHOLD:
        return [], False
    return hits[:top_k], True


def _retrieve_hybrid_no_rerank(query: str, top_k: int) -> tuple[list[SearchResult], bool]:
    """Hybrid (vector + BM25 + RRF) without cross-encoder reranking."""
    dense_hits = vector_search(query, top_k=20)
    sparse_hits = bm25_search(query, top_k=20)
    fused = reciprocal_rank_fusion(dense_hits, sparse_hits)
    # RRF scores are 1/(k+rank) — max ~0.0167, never comparable to CONFIDENCE_THRESHOLD.
    # Use presence of dense results as the context-found signal instead.
    if not fused or not dense_hits or dense_hits[0].score < CONFIDENCE_THRESHOLD:
        return [], False
    return fused[:top_k], True


def _retrieve_hybrid_with_rerank(query: str, top_k: int) -> tuple[list[SearchResult], bool]:
    """Full production pipeline: hybrid retrieval + cross-encoder reranking."""
    dense_hits = vector_search(query, top_k=20)
    sparse_hits = bm25_search(query, top_k=20)
    fused = reciprocal_rank_fusion(dense_hits, sparse_hits)
    top_results, context_found = rerank(query=query, candidates=fused, final_k=top_k)
    return top_results, context_found


# ── Single-question evaluation ───────────────────────────────────────────────

def evaluate_one(
    question_item: dict[str, Any],
    retrieval_mode: str,
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Run one question through the pipeline with the specified retrieval_mode.

    retrieval_mode: "vector_only" | "hybrid_no_rerank" | "hybrid_rerank"
    """
    q = question_item["question"]
    expected_url = question_item.get("expected_source_url", "")
    ground_truth = question_item.get("ground_truth", "")
    is_ood = question_item.get("category") == "out_of_domain"

    start = time.perf_counter()

    if retrieval_mode == "vector_only":
        top_results, context_found = _retrieve_vector_only(q, top_k)
    elif retrieval_mode == "hybrid_no_rerank":
        top_results, context_found = _retrieve_hybrid_no_rerank(q, top_k)
    else:  # hybrid_rerank (production)
        top_results, context_found = _retrieve_hybrid_with_rerank(q, top_k)

    # Generation
    if not context_found or not top_results:
        answer = NO_CONTEXT_RESPONSE
    else:
        chunk_dicts = [
            {"text": r.text, "section_heading": r.section_heading, "source_url": r.source_url}
            for r in top_results
        ]
        prompt = build_user_prompt(q, chunk_dicts)
        try:
            # Retry on 429 rate-limit errors and transient connection errors.
            import re as _re
            _max_retries = 3
            for _attempt in range(_max_retries):
                try:
                    answer = llm_client.generate(SYSTEM_PROMPT, prompt)
                    break
                except Exception as _exc:
                    _exc_str = str(_exc)
                    _is_429  = "429" in _exc_str
                    _is_conn = "Connection error" in _exc_str or "connection" in _exc_str.lower() or "timeout" in _exc_str.lower()
                    if (_is_429 or _is_conn) and _attempt < _max_retries - 1:
                        _wait = 30  # default backoff
                        if _is_429:
                            _m = _re.search(r'Please try again in (?:(\d+)m)?(\d+(?:\.\d+)?)s', _exc_str)
                            if _m:
                                _mins = int(_m.group(1)) if _m.group(1) else 0
                                _secs = float(_m.group(2))
                                _wait = (_mins * 60) + _secs + 2
                        logger.warning(
                            "%s on generation (attempt %d/%d) — sleeping %ss",
                            "Rate limit 429" if _is_429 else "Connection error",
                            _attempt + 1, _max_retries, _wait
                        )
                        print(f"       [{'Rate limit 429' if _is_429 else 'Connection error'} — sleeping {_wait:.0f}s before retry {_attempt+2}/{_max_retries}]")
                        time.sleep(_wait)
                    else:
                        raise
            else:
                answer = f"[Generation error: all {_max_retries} retries exhausted]"
        except Exception as exc:
            answer = f"[Generation error: {exc}]"

    elapsed_ms = int((time.perf_counter() - start) * 1000)

    retrieved_urls = [r.source_url for r in top_results]
    hit_at_k = expected_url in retrieved_urls if expected_url else None

    # OOD refusal correctness
    refusal_correct = is_ood and (
        not context_found or NO_CONTEXT_RESPONSE.lower() in answer.lower()
    )

    return {
        "id": question_item["id"],
        "question": q,
        "category": question_item.get("category"),
        "expected_url": expected_url,
        "ground_truth": ground_truth,
        "retrieved_urls": retrieved_urls,
        "contexts": [r.text for r in top_results],
        "hit_at_k": hit_at_k,
        "context_found": context_found,
        "refusal_correct": refusal_correct,
        "answer": answer,
        "latency_ms": elapsed_ms,
        "backend": llm_client.get_active_backend(),
    }


# ── Full eval run ─────────────────────────────────────────────────────────────

def run_eval(
    run_name: str,
    chunking: str,
    retrieval_mode: str,
    top_k: int = 5,
    skip_ragas: bool = False,
    output_name: str | None = None,
    collection_name: str = "claude_docs",
) -> dict[str, Any]:
    """
    Run all eval questions through the pipeline and compute all metrics.

    Args:
        run_name:        Human label, e.g. "Baseline" or "Final (Hybrid + Rerank)"
        chunking:        Description of chunking used, e.g. "fixed_size_naive" or "structure_aware"
        retrieval_mode:  "vector_only" | "hybrid_no_rerank" | "hybrid_rerank"
        top_k:           Number of chunks passed to LLM after retrieval/reranking
        skip_ragas:      Set True to skip RAGAS (faster, no LLM judge calls)
        output_name:     Optional filename override for results JSON
        collection_name: ChromaDB collection to query. "claude_docs" (production,
                         structure-aware) for all runs except the Baseline, which
                         uses "baseline_naive_chunks" (naive fixed-size chunking).
    """
    from src.config import RAGAS_JUDGE_BACKEND, RAGAS_JUDGE_MODEL, GROQ_MODEL

    if not EVAL_FILE.exists():
        raise FileNotFoundError(f"Eval questions missing: {EVAL_FILE}")

    with open(EVAL_FILE, encoding="utf-8") as f:
        questions = json.load(f)

    # ── Config summary — printed before every run ─────────────────────────────
    # This is the safeguard against silently repeating the last run's mistake
    # (two stages sharing the same index).
    print(f"\n{'='*65}")
    print(f"  RUN CONFIG SUMMARY")
    print(f"{'='*65}")
    print(f"  run_name     : {run_name}")
    print(f"  chunking     : {chunking}")
    print(f"  retrieval    : {retrieval_mode}")
    print(f"  reranking    : {'yes (BGE cross-encoder)' if retrieval_mode == 'hybrid_rerank' else 'no'}")
    print(f"  top_k        : {top_k}")
    print(f"  collection   : {collection_name}")
    print(f"  prod LLM     : {GROQ_MODEL}  (generation only)")
    print(f"  judge model  : {RAGAS_JUDGE_MODEL} via {RAGAS_JUDGE_BACKEND}  (scoring only, same for all 4 runs)")
    print(f"{'='*65}")

    results = []
    in_domain_hits = 0
    in_domain_total = 0
    ood_correct = 0
    ood_total = 0
    total_latency = 0

    # ── Per-question checkpoint path ──────────────────────────────────────────
    # Written after every question so a mid-run crash loses at most one result.
    # The checkpoint is a sidecar file — the canonical final JSON is written
    # once at the end (unchanged). On restart, read this file to see which
    # questions already completed.
    stamp_for_ckpt = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name_ckpt = run_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus").replace("/", "_")
    checkpoint_file = RESULTS_DIR / f"phase4_{safe_name_ckpt}_{stamp_for_ckpt}_checkpoint.json"

    with _use_collection(collection_name):
        for i, item in enumerate(questions, 1):
            print(f"  [{i:2d}/{len(questions)}] {item['id']}: {item['question'][:60]}...")
            res = evaluate_one(item, retrieval_mode=retrieval_mode, top_k=top_k)
            results.append(res)
            total_latency += res["latency_ms"]

            if res["category"] == "out_of_domain":
                ood_total += 1
                if res["refusal_correct"]:
                    ood_correct += 1
            else:
                in_domain_total += 1
                if res["hit_at_k"]:
                    in_domain_hits += 1

            # Pace generation calls: gpt-oss-120b is 8k TPM on free tier.
            # Each call uses ~2000-3000 tokens. Sleep 70s *after* the call
            # completes to let the 1-minute TPM window fully reset before the
            # next question. This avoids repeated 429 retries (each costs 2×20s).
            # OOD no-context questions skip (no LLM call made).
            _is_ood = item.get("category") == "out_of_domain"
            if res.get("context_found") and not _is_ood and not str(res.get("answer","")).startswith("[Generation error"):
                time.sleep(70)

            status = "✓" if (res["hit_at_k"] or res["refusal_correct"]) else "✗"
            print(f"       {status} context_found={res['context_found']} | latency={res['latency_ms']}ms")

            # ── Write per-question checkpoint ─────────────────────────────────
            # Overwrite after each question — worst-case crash loses one result.
            try:
                with open(checkpoint_file, "w", encoding="utf-8") as _f:
                    json.dump({
                        "run_name": run_name,
                        "collection_name": collection_name,
                        "retrieval_mode": retrieval_mode,
                        "questions_completed": i,
                        "questions_total": len(questions),
                        "results_so_far": results,
                    }, _f, indent=2)
            except OSError as _ckpt_err:
                logger.warning("Checkpoint write failed (non-fatal): %s", _ckpt_err)

    recall_at_k = in_domain_hits / in_domain_total if in_domain_total else 0.0
    ood_accuracy = ood_correct / ood_total if ood_total else 0.0
    avg_latency = int(total_latency / len(questions)) if questions else 0

    print(f"\n  Recall@{top_k}: {recall_at_k:.3f} ({in_domain_hits}/{in_domain_total})")
    print(f"  OOD Refusal Accuracy: {ood_accuracy:.3f} ({ood_correct}/{ood_total})")
    print(f"  Avg Latency: {avg_latency}ms")

    # RAGAS scoring (LLM-judge based — needs API calls)
    ragas_scores: dict[str, float] = {}
    if not skip_ragas:
        print(f"\n  Running RAGAS scoring ({len([r for r in results if r['category']!='out_of_domain' and r['context_found']])} scoreable questions)...")
        ragas_scores = compute_ragas_scores(results)
        if ragas_scores:
            for k, v in ragas_scores.items():
                print(f"    {k}: {v:.4f}")

    summary: dict[str, Any] = {
        "run_name": run_name,
        "chunking": chunking,
        "retrieval_mode": retrieval_mode,
        "collection_name": collection_name,
        "top_k": top_k,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_questions": len(questions),
        "in_domain_questions": in_domain_total,
        "ood_questions": ood_total,
        f"recall_at_{top_k}": recall_at_k,
        "ood_refusal_accuracy": ood_accuracy,
        "avg_latency_ms": avg_latency,
        **ragas_scores,
        "results": results,
    }

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = run_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus").replace("/", "_")
    # design.md §6 requires a timestamped file per stage — phase4_<stage>_<ts>.json
    out_file = RESULTS_DIR / (output_name or f"phase4_{safe_name}_{stamp}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved → {out_file.name}")

    return summary


# ── Ablation table runner ─────────────────────────────────────────────────────

def run_ablation_study(skip_ragas: bool = False) -> list[dict[str, Any]]:
    """
    Run all 4 ablation experiments and print a comparison table.

    Stage definitions (design.md §6, memory.md Phase 4 redo):

      Stage 1 — Baseline:
        Collection: baseline_naive_chunks (GENUINELY DIFFERENT from production)
        Chunking:   Naive fixed-size (512 tok, 10% overlap, no heading awareness,
                    no code-block protection) — built by build_baseline_collection.py
        Retrieval:  Vector-only (no BM25, no fusion, no reranker)

      Stage 2 — + Structure-Aware Chunking:
        Collection: claude_docs (production structure-aware index, 2,621 chunks)
        Chunking:   Structure-aware (heading splits, code-block atomic, 12% overlap)
        Retrieval:  Vector-only (no BM25, no fusion, no reranker)

      Stage 3 — + Hybrid Retrieval:
        Collection: claude_docs
        Retrieval:  Dense + BM25 + RRF fusion (no cross-encoder)

      Stage 4 — + Reranking (Final / Production):
        Collection: claude_docs
        Retrieval:  Dense + BM25 + RRF + BGE cross-encoder reranking
        This is the ONLY stage that matches the deployed production pipeline.

    The judge model (RAGAS_JUDGE_MODEL) is identical across all four stages.
    """
    runs = [
        dict(
            run_name="Baseline (naive chunks, vector-only)",
            chunking="naive_fixed_size_512tok_10pct_overlap",
            retrieval_mode="vector_only",
            collection_name="baseline_naive_chunks",
            top_k=5,
        ),
        dict(
            run_name="+ Structure-Aware Chunking",
            chunking="structure_aware_heading_split_512tok_12pct_overlap",
            retrieval_mode="vector_only",
            collection_name="claude_docs",
            top_k=5,
        ),
        dict(
            run_name="+ Hybrid Retrieval (vector + BM25 + RRF)",
            chunking="structure_aware_heading_split_512tok_12pct_overlap",
            retrieval_mode="hybrid_no_rerank",
            collection_name="claude_docs",
            top_k=5,
        ),
        dict(
            run_name="+ Reranking (Final / Production)",
            chunking="structure_aware_heading_split_512tok_12pct_overlap",
            retrieval_mode="hybrid_rerank",
            collection_name="claude_docs",
            top_k=5,
        ),
    ]

    all_summaries = []
    for cfg in runs:
        summary = run_eval(
            run_name=cfg["run_name"],
            chunking=cfg["chunking"],
            retrieval_mode=cfg["retrieval_mode"],
            collection_name=cfg["collection_name"],
            top_k=cfg["top_k"],
            skip_ragas=skip_ragas,
        )
        all_summaries.append(summary)

    # Print comparison table
    print("\n" + "=" * 90)
    print("ABLATION TABLE")
    print("=" * 90)
    header = f"{'Run':<42} {'Recall@5':>9} {'OOD Acc':>8} {'Faith':>7} {'Ans Rel':>8} {'Ctx Prec':>9} {'Ctx Rec':>8} {'Lat ms':>7}"
    print(header)
    print("-" * 90)
    for s in all_summaries:
        row = (
            f"{s['run_name']:<42} "
            f"{s.get('recall_at_5', 0):.3f}    "
            f"{s.get('ood_refusal_accuracy', 0):.3f}    "
            f"{s.get('faithfulness', float('nan')):.3f}    "
            f"{s.get('answer_relevancy', float('nan')):.3f}    "
            f"{s.get('context_precision', float('nan')):.3f}    "
            f"{s.get('context_recall', float('nan')):.3f}    "
            f"{s.get('avg_latency_ms', 0):>6}"
        )
        print(row)
    print("=" * 90)

    # Save the comparison table
    table_file = RESULTS_DIR / f"ablation_table_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(table_file, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nFull ablation results saved → {table_file.name}")

    return all_summaries


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Colophon evaluation harness")
    parser.add_argument(
        "--mode",
        choices=["ablation", "single"],
        default="ablation",
        help="ablation = run all 4 configs; single = run production config only",
    )
    parser.add_argument(
        "--skip-ragas",
        action="store_true",
        help="Skip RAGAS LLM-judge scoring (faster, no API calls for scoring)",
    )
    parser.add_argument(
        "--retrieval",
        choices=["vector_only", "hybrid_no_rerank", "hybrid_rerank"],
        default="hybrid_rerank",
        help="Retrieval mode for --mode=single",
    )
    args = parser.parse_args()

    if args.mode == "ablation":
        run_ablation_study(skip_ragas=args.skip_ragas)
    else:
        run_eval(
            run_name="Production (Hybrid + Rerank)",
            chunking="structure_aware",
            retrieval_mode=args.retrieval,
            skip_ragas=args.skip_ragas,
        )
