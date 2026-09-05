"""
Latency benchmark: 20 in-domain questions via live /query endpoint.

Uses a 60s HTTP timeout so slow answers show their real latency instead
of erroring, giving honest p50/p95 numbers.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "src" / "evaluation" / "results"
EVAL_Q = PROJECT_ROOT / "src" / "evaluation" / "eval_questions.json"
LOG = RESULTS_DIR / "latency_optionB_benchmark.log"

def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
with open(LOG, "w") as f:
    f.write("")  # clear

log("=" * 70)
log("LATENCY BENCHMARK — Option B (top-10 candidates), MPS device")
log("20 in-domain questions, /query endpoint, timeout=60s")
log("=" * 70)

with open(EVAL_Q) as f:
    all_qs = json.load(f)
in_domain_qs = [q for q in all_qs if q.get("category") != "out_of_domain"][:20]

log(f"Questions loaded: {len(in_domain_qs)} in-domain")

# Start the API server
log("Starting uvicorn server...")
server = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "src.api.main:app",
     "--port", "8001", "--host", "127.0.0.1", "--log-level", "warning"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    cwd=str(PROJECT_ROOT),
)

log("Waiting 20s for server startup (includes reranker preload on MPS)...")
time.sleep(20)

latencies: list[int] = []
errors: list[str] = []

for i, q in enumerate(in_domain_qs, 1):
    payload = json.dumps({"question": q["question"]}).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:8001/query",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        reported_ms = data.get("latency_ms", elapsed_ms)
        latencies.append(reported_ms)
        log(f"  [{i:02d}/20] {q['question'][:55]}... -> {reported_ms}ms")
    except Exception as e:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        errors.append(str(e))
        log(f"  [{i:02d}/20] ERROR after {elapsed_ms}ms: {e}")
    # Small pace to not hammer Groq
    time.sleep(5)

server.terminate()
log("Server terminated.")

log("")
log("=" * 70)
log(f"RESULTS: {len(latencies)} successful, {len(errors)} errors")
if latencies:
    s = sorted(latencies)
    p50 = s[len(s) // 2]
    p95_idx = min(int(len(s) * 0.95), len(s) - 1)
    p95 = s[p95_idx]
    p99_idx = min(int(len(s) * 0.99), len(s) - 1)
    p99 = s[p99_idx]
    min_l = s[0]
    max_l = s[-1]
    avg_l = int(sum(s) / len(s))
    log(f"  min  = {min_l}ms")
    log(f"  avg  = {avg_l}ms")
    log(f"  p50  = {p50}ms")
    log(f"  p95  = {p95}ms")
    log(f"  p99  = {p99}ms")
    log(f"  max  = {max_l}ms")
    log(f"  NFR-1 target: p95 < 6000ms  ->  {'PASS' if p95 < 6000 else 'FAIL'}")
    log(f"  All latencies: {s}")
else:
    log("  No successful measurements!")

log("=" * 70)
log("Benchmark complete.")
