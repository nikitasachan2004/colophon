"""
Stage 4 RAGAS re-run with Option B capping (top-10 rerank candidates).

Runs the full hybrid+rerank pipeline with RERANK_CANDIDATE_POOL=10
to validate that faithfulness/precision/recall haven't regressed
after the candidate capping change.
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Force offline to use cached HF models
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

# Pin judge and generator
os.environ["RAGAS_JUDGE_BACKEND"] = "ollama"
os.environ["RAGAS_JUDGE_MODEL"] = "llama3.1:8b"
os.environ["RAGAS_JUDGE_OLLAMA_URL"] = "http://localhost:11434"
os.environ["GROQ_MODEL"] = "openai/gpt-oss-120b"
# Confirm candidate pool capping is active
os.environ["RERANK_CANDIDATE_POOL"] = "10"
os.environ["FORCE_CPU_RERANKER"] = "1"

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*langchain_community.*")

from dotenv import load_dotenv
load_dotenv()

import src.config as _cfg
import src.generation.groq_backend as _groq_be
_cfg.GROQ_MODEL = "openai/gpt-oss-120b"
_groq_be.GROQ_MODEL = "openai/gpt-oss-120b"
_cfg.RAGAS_JUDGE_MODEL = "llama3.1:8b"
_cfg.RAGAS_JUDGE_BACKEND = "ollama"
_cfg.RERANK_CANDIDATE_POOL = 10

from src.evaluation.run_eval import run_eval, compute_ragas_scores, RESULTS_DIR

GEN_MODEL   = "openai/gpt-oss-120b"
JUDGE_MODEL = "llama3.1:8b via ollama (local)"

print(f"\n{'='*72}")
print(f"  STAGE 4 RAGAS RE-RUN — Option B (RERANK_CANDIDATE_POOL=10)")
print(f"{'='*72}")
print(f"  Judge  : {JUDGE_MODEL}")
print(f"  Gen LLM: {GEN_MODEL} (Groq)")
print(f"  Pool   : RERANK_CANDIDATE_POOL=10 (Option B)")
print(f"{'='*72}\n")

print("Running Stage 4 question generation...")
summary = run_eval(
    run_name="+ Reranking (Final / Production) [OptionB-pool10]",
    chunking="structure_aware_heading_split_512tok_12pct_overlap",
    retrieval_mode="hybrid_rerank",
    collection_name="claude_docs",
    top_k=5,
    skip_ragas=True,
)
results = summary["results"]
print(f"Questions done: Recall@5={summary.get('recall_at_5', summary.get('recall_at_k', 0)):.3f}")

errors = sum(1 for r in results if str(r.get("answer", "")).startswith("[Generation error"))
if errors:
    print(f"ERROR: {errors} generation errors. Aborting before RAGAS scoring to avoid corrupting results.")
    sys.exit(1)

print("\nRunning RAGAS scoring...")
ragas_scores = compute_ragas_scores(results)
print(f"RAGAS scores: {ragas_scores}")

in_hits  = sum(1 for r in results if r.get("hit_at_k") and r.get("category") != "out_of_domain")
in_total = sum(1 for r in results if r.get("category") != "out_of_domain")
ood_ok   = sum(1 for r in results if r.get("refusal_correct"))
ood_tot  = sum(1 for r in results if r.get("category") == "out_of_domain")
avg_lat  = int(sum(r.get("latency_ms", 0) for r in results) / len(results)) if results else 0

final_summary = {
    "run_name": "+ Reranking (Final / Production) [OptionB-pool10]",
    "chunking": "structure_aware_heading_split_512tok_12pct_overlap",
    "retrieval_mode": "hybrid_rerank",
    "rerank_candidate_pool": 10,
    "collection_name": "claude_docs",
    "top_k": 5,
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
out_file = RESULTS_DIR / f"phase4_plus_reranking_optionB_pool10_{stamp}.json"
with open(out_file, "w") as f:
    json.dump(final_summary, f, indent=2)
print(f"\nSaved: {out_file.name}")

print(f"\n{'='*80}")
print("STAGE 4 OPTION B — FINAL METRICS")
print(f"{'='*80}")
print(f"  Recall@5           : {final_summary['recall_at_5']:.3f}")
print(f"  OOD Refusal Acc    : {final_summary['ood_refusal_accuracy']:.3f}")
print(f"  Faithfulness       : {ragas_scores.get('faithfulness', float('nan')):.4f}")
print(f"  Answer Relevancy   : {ragas_scores.get('answer_relevancy', float('nan')):.4f}")
print(f"  Context Precision  : {ragas_scores.get('context_precision', float('nan')):.4f}")
print(f"  Context Recall     : {ragas_scores.get('context_recall', float('nan')):.4f}")
print(f"  Avg Latency        : {avg_lat}ms")
print(f"{'='*80}")
print("Done.")
