"""Stage 3 RAGAS scoring — runs only if generation file is clean."""
import os, sys, json, warnings, glob
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ["RAGAS_JUDGE_BACKEND"] = "ollama"
os.environ["RAGAS_JUDGE_MODEL"] = "llama3.1:8b"
os.environ["RAGAS_JUDGE_OLLAMA_URL"] = "http://localhost:11434"
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*langchain_community.*")
sys.path.insert(0, ".")
from dotenv import load_dotenv; load_dotenv()

from src.evaluation.run_eval import compute_ragas_scores, RESULTS_DIR

# Find the latest Stage 3 clean generation file (no INVALID suffix)
files = sorted(f for f in RESULTS_DIR.glob("phase4_plus_hybrid_retrieval_*.json")
               if "INVALID" not in f.name)
if not files:
    print("No Stage 3 file found."); sys.exit(1)

latest = files[-1]
print(f"Loading: {latest.name}")
with open(latest) as f:
    data = json.load(f)

results = data.get("results", [])
errors = sum(1 for r in results if str(r.get("answer","")).startswith("[Generation error"))
if errors > 0:
    print(f"ERROR: {errors} generation errors — cannot score RAGAS on corrupted data.")
    sys.exit(1)

print(f"0 generation errors confirmed. Starting RAGAS (Ollama llama3.1:8b, max_workers=1)...")
scores = compute_ragas_scores(results)

print("\n--- STAGE 3 RAGAS SCORES ---")
for k, v in sorted(scores.items()):
    print(f"  {k}: {v:.4f}")

# Also print per-question context check vs Stage 2
with open("src/evaluation/results/phase4_plus_structure-aware_chunking_20260903_140340.json") as f:
    d2 = json.load(f)
by_id2 = {r["id"]: r for r in d2["results"]}
in_domain3 = [r for r in results if r.get("category") != "out_of_domain" and r.get("context_found") and r.get("contexts")]

same_count = 0
diff_ids = []
for r in in_domain3:
    r2 = by_id2.get(r["id"])
    if r2 and r.get("contexts") == r2.get("contexts"):
        same_count += 1
    elif r2:
        diff_ids.append(r["id"])

print(f"\n--- CONTEXT IDENTITY vs STAGE 2 ---")
print(f"  {same_count}/{len(in_domain3)} in-domain questions: identical contexts to Stage 2")
print(f"  {len(diff_ids)} questions with genuinely different contexts: {diff_ids}")

# Save scores into the file
data.update(scores)
with open(latest, "w") as f:
    json.dump(data, f, indent=2)
print(f"\nSaved scores → {latest.name}")
