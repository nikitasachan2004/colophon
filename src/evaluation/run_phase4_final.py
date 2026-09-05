"""
Phase 4 final runner (Sep 2026).

Executes Stages 2, 3, and 4 cleanly using current connection-error retry logic.
Reuses the clean Stage 1 Baseline result.
Computes RAGAS for Stages 2, 3, 4 sequentially using local Ollama.
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

os.environ["RAGAS_JUDGE_BACKEND"] = "ollama"
os.environ["RAGAS_JUDGE_MODEL"] = "llama3.1:8b"
os.environ["RAGAS_JUDGE_OLLAMA_URL"] = "http://localhost:11434"
os.environ["GROQ_MODEL"] = "openai/gpt-oss-120b"

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*langchain_community.*")

from dotenv import load_dotenv
load_dotenv()

from src.evaluation.run_eval import run_eval, compute_ragas_scores, RESULTS_DIR

import src.config as _cfg
import src.generation.groq_backend as _groq_be
_cfg.GROQ_MODEL = "openai/gpt-oss-120b"
_groq_be.GROQ_MODEL = "openai/gpt-oss-120b"
_cfg.RAGAS_JUDGE_MODEL = "llama3.1:8b"
_cfg.RAGAS_JUDGE_BACKEND = "ollama"

GEN_MODEL   = "openai/gpt-oss-120b"
JUDGE_MODEL = "llama3.1:8b via ollama (local)"

print(f"\\n{'='*72}")
print(f"  PHASE 4 FINAL RUNNER (STAGES 2, 3, 4 + RAGAS)")
print(f"{'='*72}")
print(f"  Judge  : {JUDGE_MODEL}")
print(f"  Gen LLM: {GEN_MODEL} (Groq, connection retry fixed)")
print(f"{'='*72}\\n")

STAGES = [
    dict(
        stage_num=2,
        run_name="+ Structure-Aware Chunking",
        chunking="structure_aware_heading_split_512tok_12pct_overlap",
        retrieval_mode="vector_only",
        collection_name="claude_docs",
        top_k=5,
    ),
    dict(
        stage_num=3,
        run_name="+ Hybrid Retrieval (vector + BM25 + RRF)",
        chunking="structure_aware_heading_split_512tok_12pct_overlap",
        retrieval_mode="hybrid_no_rerank",
        collection_name="claude_docs",
        top_k=5,
    ),
    dict(
        stage_num=4,
        run_name="+ Reranking (Final / Production)",
        chunking="structure_aware_heading_split_512tok_12pct_overlap",
        retrieval_mode="hybrid_rerank",
        collection_name="claude_docs",
        top_k=5,
    ),
]

# Get stage 1 baseline results
STAGE1_PATH = RESULTS_DIR / "phase4_baseline_naive_chunks,_vector-only_20260902_204914.json"
with open(STAGE1_PATH) as f:
    stage1_summary = json.load(f)

print("\\n" + "="*72)
print("  PHASE A — QUESTION EXECUTION (Stages 2, 3, 4)")
print("="*72)

stage_results = {1: stage1_summary.get("results", [])}

for stage in STAGES:
    snum = stage["stage_num"]
    print(f"\\n{'─'*72}")
    print(f"  Stage {snum}/4 questions: {stage['run_name']}")
    print(f"{'─'*72}")

    try:
        # Run generation cleanly
        summary = run_eval(
            run_name=stage["run_name"],
            chunking=stage["chunking"],
            retrieval_mode=stage["retrieval_mode"],
            collection_name=stage["collection_name"],
            top_k=stage["top_k"],
            skip_ragas=True, 
        )
        stage_results[snum] = summary["results"]
        stage["_summary"] = summary
        print(f"  → Questions done: Recall@5={summary.get('recall_at_5', summary.get('recall_at_k', 0)):.3f}")
    except Exception as exc:
        print(f"  ✗ Stage {snum} questions FAILED: {exc}")
        import traceback; traceback.print_exc()
        stage_results[snum] = []
        stage["_error"] = str(exc)

print("\\n\\n" + "="*72)
print("  PHASE B — RAGAS SCORING (Stages 2, 3, 4)")
print("="*72)

all_summaries = [stage1_summary]

for stage in STAGES:
    snum = stage["stage_num"]
    results = stage_results.get(snum, [])
    print(f"\\n  Stage {snum}/4 RAGAS: {stage['run_name']}")

    if not results:
        print(f"  → No results for stage {snum}, skipping RAGAS")
        all_summaries.append({"run_name": stage["run_name"], "error": "no results"})
        continue

    errors = sum(1 for r in results if str(r.get("answer","")).startswith("[Generation error"))
    if errors > 0:
        print(f"  WARNING: {errors} generation errors — RAGAS scores may be degraded")

    ragas_scores = compute_ragas_scores(results)
    print(f"  → RAGAS done: {ragas_scores}")

    in_hits  = sum(1 for r in results if r.get("hit_at_k") and r.get("category") != "out_of_domain")
    in_total = sum(1 for r in results if r.get("category") != "out_of_domain")
    ood_ok   = sum(1 for r in results if r.get("refusal_correct"))
    ood_tot  = sum(1 for r in results if r.get("category") == "out_of_domain")
    avg_lat  = int(sum(r.get("latency_ms", 0) for r in results) / len(results)) if results else 0

    summary = {
        "run_name": stage["run_name"],
        "chunking": stage["chunking"],
        "retrieval_mode": stage["retrieval_mode"],
        "collection_name": stage["collection_name"],
        "top_k": stage["top_k"],
        "judge_model": JUDGE_MODEL,
        "gen_model": GEN_MODEL,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_questions": len(results),
        "in_domain_questions": in_total,
        "ood_questions": ood_tot,
        "recall_at_5": in_hits / in_total if in_total else 0.0,
        "ood_refusal_accuracy": ood_ok / ood_tot if ood_tot else 0.0,
        "avg_latency_ms": avg_lat,
        **ragas_scores,
        "results": results,
    }

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = stage["run_name"].lower().replace(" ","_").replace("(","").replace(")","").replace("+","plus").replace("/","_")
    out_file = RESULTS_DIR / f"phase4_{safe}_clean_{stamp}.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  → Saved: {out_file.name}")
    all_summaries.append(summary)

print(f"\\n\\n{'='*120}")
print(f"  ABLATION COMPARISON TABLE  judge={JUDGE_MODEL}  gen={GEN_MODEL}  {datetime.now():%Y-%m-%d}")
print(f"{'='*120}")
hdr = (f"{'Stage':<44} {'Recall@5':>8} {'OOD Acc':>8}"
       f" {'Faith':>8} {'AnsRel':>8} {'CtxPrec':>8} {'CtxRec':>8} {'AvgLat(ms)':>11}")
print(hdr)
print("-" * 120)
for r in all_summaries:
    if "error" in r and "run_name" in r:
        print(f"  {r['run_name']:<42}  ERROR: {r.get('error')}")
        continue
    row = (
        f"  {r['run_name']:<42}"
        f"  {r.get('recall_at_5', float('nan')):>8.3f}"
        f"  {r.get('ood_refusal_accuracy', float('nan')):>8.3f}"
        f"  {r.get('faithfulness', float('nan')):>8.4f}"
        f"  {r.get('answer_relevancy', float('nan')):>8.4f}"
        f"  {r.get('context_precision', float('nan')):>8.4f}"
        f"  {r.get('context_recall', float('nan')):>8.4f}"
        f"  {r.get('avg_latency_ms', 0):>11d}"
    )
    print(row)
print(f"{'='*120}")
print("\\nDone.")
