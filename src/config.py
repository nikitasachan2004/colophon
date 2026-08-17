"""
Central configuration — single source of truth for all env-driven parameters.

Why this exists: rules.md §4 forbids hardcoded magic numbers for tunable parameters.
Every chunking, retrieval, and threshold constant lives here and is driven by .env,
so Phase 4 tuning requires zero code changes.
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", str(DATA_DIR / "chroma_db"))

# ── LLM Backend ───────────────────────────────────────────────────────
LLM_BACKEND: str = os.getenv("LLM_BACKEND", "groq")  # "ollama" | "groq"

# Ollama (local dev)
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

# Groq (production)
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# ── RAGAS Evaluation Judge — deliberately separate from production path ─
#
# Design decision (memory.md Phase 4 redo): the RAGAS LLM judge must be
# completely independent of LLM_BACKEND and GROQ_MODEL, which control the
# production RAG generation pipeline. Using the same model for both caused
# the last ablation run to drift across three different judge models as
# Groq's per-model daily quota was exhausted mid-run — making all four
# stages' RAGAS scores non-comparable to each other.
#
# The judge runs on Ollama locally (RAGAS_JUDGE_BACKEND=ollama), which:
#   1. Has zero quota — no rate limits, no TPD cap, no mid-run exhaustion.
#   2. Is fixed to the same model for all four ablation runs in one session.
#   3. Is never used for production answer generation — only for scoring.
#   4. Keeps evaluation fully offline — no external API calls for the judge.
#
# Fallback: if Ollama is unavailable, set RAGAS_JUDGE_BACKEND=groq and
# RAGAS_JUDGE_MODEL=allam-2-7b (smallest Groq model, its own quota bucket,
# separate from both production models gpt-oss-120b and gpt-oss-20b).
RAGAS_JUDGE_BACKEND: str = os.getenv("RAGAS_JUDGE_BACKEND", "ollama")  # "ollama" | "groq"
RAGAS_JUDGE_MODEL: str = os.getenv("RAGAS_JUDGE_MODEL", "llama3.1:8b")
RAGAS_JUDGE_OLLAMA_URL: str = os.getenv("RAGAS_JUDGE_OLLAMA_URL", "http://localhost:11434")

# ── Embeddings ─────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# ── Chunking (tuned in Phase 4 — design.md §3) ────────────────────────
CHUNK_SIZE_TOKENS: int = int(os.getenv("CHUNK_SIZE_TOKENS", "512"))
CHUNK_OVERLAP_PCT: float = float(os.getenv("CHUNK_OVERLAP_PCT", "0.12"))

# ── Retrieval ──────────────────────────────────────────────────────────
RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "20"))
RERANK_CANDIDATE_POOL: int = int(os.getenv("RERANK_CANDIDATE_POOL", "10"))
RERANK_FINAL_K: int = int(os.getenv("RERANK_FINAL_K", "5"))
CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.35"))

# ── API ────────────────────────────────────────────────────────────────
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))
ENABLE_ADMIN: bool = os.getenv("ENABLE_ADMIN", "false").lower() == "true"

# ── Target corpus: Anthropic Claude API Documentation ─────────────────
# Canonical domain: platform.claude.com/docs (docs.claude.com redirects here via 301)
TARGET_DOC_URLS: list[str] = [
    # Getting Started & Core Messages API
    "https://platform.claude.com/docs/en/intro",
    "https://platform.claude.com/docs/en/get-api-key",
    "https://platform.claude.com/docs/en/get-started",
    "https://platform.claude.com/docs/en/manage-claude/authentication",
    "https://platform.claude.com/docs/en/build-with-claude/overview",
    "https://platform.claude.com/docs/en/build-with-claude/working-with-messages",
    "https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons",
    "https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback",
    "https://platform.claude.com/docs/en/build-with-claude/structured-outputs",
    "https://platform.claude.com/docs/en/build-with-claude/citations",
    "https://platform.claude.com/docs/en/build-with-claude/streaming",
    "https://platform.claude.com/docs/en/build-with-claude/batch-processing",
    "https://platform.claude.com/docs/en/build-with-claude/prompt-caching",
    "https://platform.claude.com/docs/en/build-with-claude/embeddings",

    # Extended Thinking & Reasoning
    "https://platform.claude.com/docs/en/build-with-claude/extended-thinking",
    "https://platform.claude.com/docs/en/build-with-claude/thinking",
    "https://platform.claude.com/docs/en/build-with-claude/thinking-steering-and-cost",
    "https://platform.claude.com/docs/en/build-with-claude/thinking-tool-workflows",
    "https://platform.claude.com/docs/en/build-with-claude/thinking-troubleshooting",

    # Tool Use & Function Calling
    "https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview",
    "https://platform.claude.com/docs/en/agents-and-tools/tool-use/how-tool-use-works",
    "https://platform.claude.com/docs/en/agents-and-tools/tool-use/build-a-tool-using-agent",
    "https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools",
    "https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls",
    "https://platform.claude.com/docs/en/agents-and-tools/tool-use/parallel-tool-use",
    "https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-runner",
    "https://platform.claude.com/docs/en/agents-and-tools/tool-use/strict-tool-use",
    "https://platform.claude.com/docs/en/agents-and-tools/tool-use/server-tools",
    "https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool",
    "https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-fetch-tool",
    "https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool",
    "https://platform.claude.com/docs/en/agents-and-tools/tool-use/advisor-tool",
    "https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-use-with-prompt-caching",

    # Model Context Protocol (MCP) & Agent Skills
    "https://platform.claude.com/docs/en/agents-and-tools/mcp-connector",
    "https://platform.claude.com/docs/en/agents-and-tools/remote-mcp-servers",
    "https://platform.claude.com/docs/en/agents-and-tools/mcp-tunnels/overview",
    "https://platform.claude.com/docs/en/agents-and-tools/mcp-tunnels/concepts",
    "https://platform.claude.com/docs/en/agents-and-tools/mcp-tunnels/security",
    "https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview",
    "https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices",
    "https://platform.claude.com/docs/en/agents-and-tools/agent-skills/quickstart",
    "https://platform.claude.com/docs/en/agents-and-tools/agent-skills/enterprise",

    # Vision & Multimodal
    "https://platform.claude.com/docs/en/build-with-claude/vision",
    "https://platform.claude.com/docs/en/build-with-claude/files",
    "https://platform.claude.com/docs/en/build-with-claude/multilingual-support",

    # Models, Pricing & Token Limits
    "https://platform.claude.com/docs/en/about-claude/models/choosing-a-model",
    "https://platform.claude.com/docs/en/about-claude/models/model-ids-and-versions",
    "https://platform.claude.com/docs/en/about-claude/models/optimizing-for-cost-and-intelligence",
    "https://platform.claude.com/docs/en/about-claude/pricing",
    "https://platform.claude.com/docs/en/about-claude/model-deprecations",

    # Prompt Engineering & Optimization
    "https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview",
    "https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices",
    "https://platform.claude.com/docs/en/test-and-evaluate/develop-tests",
    "https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/reduce-latency",
    "https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/reduce-hallucinations",
    "https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/increase-consistency",

    # API Errors & Rate Limits
    "https://platform.claude.com/docs/en/api/errors",
    "https://platform.claude.com/docs/en/api/rate-limits",

    # API Reference
    "https://platform.claude.com/docs/en/api/messages",
    "https://platform.claude.com/docs/en/api/messages/create",
    "https://platform.claude.com/docs/en/api/messages/count_tokens",
    "https://platform.claude.com/docs/en/api/messages/batches",
    "https://platform.claude.com/docs/en/api/messages/batches/create",
    "https://platform.claude.com/docs/en/api/messages/batches/retrieve",
    "https://platform.claude.com/docs/en/api/messages/batches/results",
    "https://platform.claude.com/docs/en/api/messages/batches/list",
    "https://platform.claude.com/docs/en/api/messages/batches/cancel",
    "https://platform.claude.com/docs/en/api/messages/batches/delete",
    "https://platform.claude.com/docs/en/api/models",
    "https://platform.claude.com/docs/en/api/models/list",
    "https://platform.claude.com/docs/en/api/models/retrieve",
    "https://platform.claude.com/docs/en/api/files",
    "https://platform.claude.com/docs/en/api/files/upload",
    "https://platform.claude.com/docs/en/api/files/list",
    "https://platform.claude.com/docs/en/api/files/retrieve_metadata",
    "https://platform.claude.com/docs/en/api/files/download",
    "https://platform.claude.com/docs/en/api/files/delete",
]

# ── Ensure data dirs exist ─────────────────────────────────────────────
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
