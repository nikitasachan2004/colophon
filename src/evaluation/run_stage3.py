"""
Stage 3 clean generation run — Hybrid Retrieval (Vector + BM25 + RRF), no reranking.
Stops after generation and reports error count. RAGAS run separately.
"""
import os, sys, warnings
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ["RAGAS_JUDGE_BACKEND"] = "ollama"
os.environ["RAGAS_JUDGE_MODEL"] = "llama3.1:8b"
os.environ["RAGAS_JUDGE_OLLAMA_URL"] = "http://localhost:11434"
os.environ["GROQ_MODEL"] = "openai/gpt-oss-120b"
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*langchain_community.*")
sys.path.insert(0, ".")
from dotenv import load_dotenv; load_dotenv()

import src.config as _cfg
import src.generation.groq_backend as _groq_be
_cfg.GROQ_MODEL = "openai/gpt-oss-120b"
_groq_be.GROQ_MODEL = "openai/gpt-oss-120b"

from src.evaluation.run_eval import run_eval

print("Running Stage 3 Questions: + Hybrid Retrieval (vector + BM25 + RRF)...")
summary = run_eval(
    run_name="+ Hybrid Retrieval (vector + BM25 + RRF)",
    chunking="structure_aware_heading_split_512tok_12pct_overlap",
    retrieval_mode="hybrid_no_rerank",
    collection_name="claude_docs",
    top_k=5,
    skip_ragas=True,
)

results = summary["results"]
errors = [r for r in results if str(r.get("answer", "")).startswith("[Generation error")]
ood = [r for r in results if r.get("category") == "out_of_domain"]
ood_correct = [r for r in ood if r.get("refusal_correct")]

print(f"\n{'='*60}")
print(f"  STAGE 3 GENERATION COMPLETE")
print(f"{'='*60}")
print(f"  Total questions : {len(results)}")
print(f"  Generation errors: {len(errors)}")
if errors:
    for e in errors:
        print(f"    ✗ {e['id']}: {str(e.get('answer',''))[:80]}")
print(f"  OOD refusal accuracy: {len(ood_correct)}/{len(ood)}")
for r in ood:
    status = "✓ refused" if r.get("refusal_correct") else "✗ FAILED"
    ctx = r.get("context_found", False)
    print(f"    {r['id']} ({r['question'][:50]}): {status}  context_found={ctx}")
print(f"  Recall@5: {summary.get('recall_at_5', 0):.3f}")
print(f"{'='*60}")
if len(errors) == 0:
    print("  ✓ CLEAN — proceed to RAGAS scoring")
else:
    print(f"  ✗ {len(errors)} errors — DO NOT run RAGAS on this data")
