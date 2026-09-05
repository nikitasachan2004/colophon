import os
import sys
import json
import warnings
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

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*langchain_community.*")

from dotenv import load_dotenv
load_dotenv()

from src.evaluation.run_eval import compute_ragas_scores, RESULTS_DIR

# Find the latest stage 2 output file (the one just generated)
files = sorted(RESULTS_DIR.glob("phase4_plus_structure-aware_chunking_*.json"))
latest_file = files[-1] if files else None

if not latest_file:
    print("No Stage 2 file found.")
    sys.exit(1)

print(f"Loading {latest_file.name} for RAGAS scoring...")
with open(latest_file) as f:
    data = json.load(f)

results = data.get("results", [])
errors = sum(1 for r in results if str(r.get("answer", "")).startswith("[Generation error"))
if errors > 0:
    print(f"Error: {errors} generation errors found in the file! Cannot proceed with RAGAS.")
    sys.exit(1)

print("Starting RAGAS scoring using local Ollama (llama3.1:8b)...")
scores = compute_ragas_scores(results)

print("\n--- RAGAS SCORES ---")
for k, v in scores.items():
    print(f"{k}: {v:.4f}")

# Update the file with the scores
data.update(scores)
with open(latest_file, "w") as f:
    json.dump(data, f, indent=2)

print("\nSaved scores to file.")
