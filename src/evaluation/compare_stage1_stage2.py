"""
Per-question context_precision and context_recall comparison for Stage 1 vs Stage 2.

This script:
1. Loads Stage 1 and Stage 2 result files
2. Prints the actual contexts side by side for all 20 in-domain questions
3. Prints the code path that produces the RAGAS dataset to confirm no shared state
4. Re-runs RAGAS on BOTH files independently and captures per-row scores
"""
import os, sys, json, warnings
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ["RAGAS_JUDGE_BACKEND"] = "ollama"
os.environ["RAGAS_JUDGE_MODEL"] = "llama3.1:8b"
os.environ["RAGAS_JUDGE_OLLAMA_URL"] = "http://localhost:11434"
warnings.filterwarnings("ignore", category=DeprecationWarning)
sys.path.insert(0, ".")
from dotenv import load_dotenv; load_dotenv()

with open("src/evaluation/results/phase4_baseline_naive_chunks,_vector-only_20260902_204914.json") as f:
    d1 = json.load(f)
with open("src/evaluation/results/phase4_plus_structure-aware_chunking_20260903_140340.json") as f:
    d2 = json.load(f)

in_domain1 = [r for r in d1["results"] if r["category"] != "out_of_domain" and r.get("context_found") and r.get("contexts")]
in_domain2 = [r for r in d2["results"] if r["category"] != "out_of_domain" and r.get("context_found") and r.get("contexts")]

by_id1 = {r["id"]: r for r in in_domain1}
by_id2 = {r["id"]: r for r in in_domain2}
shared_ids = sorted(set(by_id1) & set(by_id2))

print(f"\n{'='*110}")
print(f"  STEP 1: Per-question context IDENTITY check (first 80 chars per chunk)")
print(f"{'='*110}")
print(f"{'ID':<12} {'Contexts identical?':<22} {'Stage1 chunk1[:60]':<62} {'Stage2 chunk1[:60]'}")
print(f"{'-'*110}")
identical_count = 0
for qid in shared_ids:
    r1 = by_id1[qid]
    r2 = by_id2[qid]
    identical = r1["contexts"] == r2["contexts"]
    if identical:
        identical_count += 1
    s1_c1 = repr(r1["contexts"][0][:60])
    s2_c1 = repr(r2["contexts"][0][:60])
    flag = "✓ SAME" if identical else "✗ DIFFERENT"
    print(f"  {qid:<10} {flag:<22} {s1_c1:<62} {s2_c1}")

print(f"\n  {identical_count}/{len(shared_ids)} in-domain questions have byte-for-byte identical contexts")

print(f"\n\n{'='*110}")
print(f"  STEP 2: Re-run RAGAS on Stage 1 and Stage 2 independently, capture per-row scores")
print(f"{'='*110}")

MAX_CTX_CHARS = 300

def build_dataset_and_score(rows, label):
    from ragas import evaluate
    from ragas.metrics import context_precision, context_recall
    from ragas.run_config import RunConfig
    from datasets import Dataset
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    import src.ingestion.embedder as emb_mod
    from langchain_community.embeddings import HuggingFaceEmbeddings

    print(f"\n  [{label}] Building dataset from {len(rows)} rows, id(rows)={id(rows)}")
    dataset = Dataset.from_dict({
        "question":     [r["question"] for r in rows],
        "answer":       [r["answer"] for r in rows],
        "contexts":     [[c[:MAX_CTX_CHARS] for c in r["contexts"]] for r in rows],
        "ground_truth": [r["ground_truth"] for r in rows],
    })
    print(f"  [{label}] dataset id={id(dataset)}, first question: {rows[0]['question'][:60]}")

    from langchain_ollama import ChatOllama
    from langchain_community.embeddings import HuggingFaceEmbeddings
    llm_raw = ChatOllama(base_url="http://localhost:11434", model="llama3.1:8b", temperature=0)
    emb_raw = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llm = LangchainLLMWrapper(langchain_llm=llm_raw)
    emb = LangchainEmbeddingsWrapper(embeddings=emb_raw)

    result = evaluate(
        dataset,
        metrics=[context_precision, context_recall],
        llm=llm,
        embeddings=emb,
        raise_exceptions=False,
        show_progress=True,
        run_config=RunConfig(max_workers=1, timeout=300),
    )
    df = result.to_pandas()
    print(f"  [{label}] result id={id(result)}, df id={id(df)}")
    return df

df1 = build_dataset_and_score(in_domain1, "STAGE1")
df2 = build_dataset_and_score(in_domain2, "STAGE2")

# Print side-by-side table
print(f"\n\n{'='*110}")
print(f"  STEP 3: Per-row comparison table — context_precision and context_recall")
print(f"{'='*110}")
print(f"{'ID':<12} {'S1_ctx_prec':>12} {'S2_ctx_prec':>12} {'Match_prec':>11} | {'S1_ctx_rec':>11} {'S2_ctx_rec':>11} {'Match_rec':>10}")
print(f"{'-'*110}")

import numpy as np
prec_diffs = []
rec_diffs = []
for i, qid in enumerate(shared_ids):
    s1_cp = df1.loc[i, "context_precision"] if "context_precision" in df1.columns else float("nan")
    s1_cr = df1.loc[i, "context_recall"]    if "context_recall"    in df1.columns else float("nan")
    s2_cp = df2.loc[i, "context_precision"] if "context_precision" in df2.columns else float("nan")
    s2_cr = df2.loc[i, "context_recall"]    if "context_recall"    in df2.columns else float("nan")

    cp_match = "✓" if (np.isnan(s1_cp) and np.isnan(s2_cp)) or abs(s1_cp - s2_cp) < 1e-6 else f"Δ={s2_cp-s1_cp:+.4f}"
    cr_match = "✓" if (np.isnan(s1_cr) and np.isnan(s2_cr)) or abs(s1_cr - s2_cr) < 1e-6 else f"Δ={s2_cr-s1_cr:+.4f}"
    prec_diffs.append(abs(s2_cp - s1_cp) if not (np.isnan(s1_cp) or np.isnan(s2_cp)) else 0.0)
    rec_diffs.append(abs(s2_cr - s1_cr) if not (np.isnan(s1_cr) or np.isnan(s2_cr)) else 0.0)

    print(f"  {qid:<10} {s1_cp:>12.4f} {s2_cp:>12.4f} {str(cp_match):>11} | {s1_cr:>11.4f} {s2_cr:>11.4f} {str(cr_match):>10}")

print(f"{'-'*110}")
print(f"  {'nanmean':<10} {np.nanmean(df1['context_precision']):>12.4f} {np.nanmean(df2['context_precision']):>12.4f}"
      f"               | {np.nanmean(df1['context_recall']):>11.4f} {np.nanmean(df2['context_recall']):>11.4f}")
print(f"\n  Max per-question prec delta: {max(prec_diffs):.6f}")
print(f"  Max per-question rec  delta: {max(rec_diffs):.6f}")
