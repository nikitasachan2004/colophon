"""
Phase 4 cleanup pass — re-run Stages 2, 3, 4 questions with connection-error
retry (fixed in run_eval.py), then re-score RAGAS on clean answers.

Stage 1 is already clean (0 errors, Recall@5=0.800, full RAGAS scored).
Stage 1 result: phase4_baseline_naive_chunks,_vector-only_20260902_204914.json
"""
from __future__ import annotations

import json, os, sys, time, warnings
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

os.environ["RAGAS_JUDGE_BACKEND"] = "ollama"
os.environ["RAGAS_JUDGE_MODEL"]   = "llama3.1:8b"
os.environ["RAGAS_JUDGE_OLLAMA_URL"] = "http://localhost:11434"
os.environ["GROQ_MODEL"]          = "openai/gpt-oss-120b"

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*langchain_community.*")

from dotenv import load_dotenv; load_dotenv()
from src.evaluation.run_eval import run_eval, compute_ragas_scores, RESULTS_DIR

import src.config as _cfg
import src.generation.groq_backend as _groq_be
_cfg.GROQ_MODEL = "openai/gpt-oss-120b"
_groq_be.GROQ_MODEL = "openai/gpt-oss-120b"
_cfg.RAGAS_JUDGE_MODEL   = "llama3.1:8b"
_cfg.RAGAS_JUDGE_BACKEND = "ollama"

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

print("\n" + "="*72)
print("  PHASE 4 CLEANUP — Re-run Stages 2-4 (connection-error retry fixed)")
print("="*72)
print("  Judge  : llama3.1:8b via ollama  (local, zero quota)")
print("  Gen LLM: openai/gpt-oss-120b  (with connection-error retry)")
print("="*72 + "\n")

for stage in STAGES:
    print(f"\n{'#'*60}")
    print(f"  Stage {stage['stage_num']}/4: {stage['run_name']}")
    print(f"{'#'*60}")
    try:
        summary = run_eval(
            run_name=stage["run_name"],
            chunking=stage["chunking"],
            retrieval_mode=stage["retrieval_mode"],
            collection_name=stage["collection_name"],
            top_k=stage["top_k"],
            skip_ragas=False,
        )
        errors = sum(1 for r in summary.get("results",[]) if str(r.get("answer","")).startswith("[Generation error"))
        print(f"  → Complete: Recall@5={summary.get('recall_at_5',0):.3f}  OOD={summary.get('ood_refusal_accuracy',0):.3f}  gen_errors={errors}")
    except Exception as exc:
        print(f"  ✗ Failed: {exc}")
        import traceback; traceback.print_exc()

print("\nCleanup done.")
