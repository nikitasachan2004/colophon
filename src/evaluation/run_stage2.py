import os
import sys
import json
import warnings
from datetime import datetime

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

from src.evaluation.run_eval import run_eval, RESULTS_DIR
import src.config as _cfg
import src.generation.groq_backend as _groq_be

_cfg.GROQ_MODEL = "openai/gpt-oss-120b"
_groq_be.GROQ_MODEL = "openai/gpt-oss-120b"

print("Running Stage 2 Questions...")
try:
    summary = run_eval(
        run_name="+ Structure-Aware Chunking",
        chunking="structure_aware_heading_split_512tok_12pct_overlap",
        retrieval_mode="vector_only",
        collection_name="claude_docs",
        top_k=5,
        skip_ragas=True,
    )
    
    results = summary["results"]
    errors = sum(1 for r in results if str(r.get("answer", "")).startswith("[Generation error"))
    print(f"\nStage 2 completed. Generation errors: {errors}")
except Exception as e:
    print(f"Error running Stage 2: {e}")
