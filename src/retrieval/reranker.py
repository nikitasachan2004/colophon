"""
Cross-encoder reranker — the precision pass after hybrid retrieval.

Why: bi-encoder/BM25 retrieval is cheap but only "topically similar." A cross-encoder
sees query+chunk together, which is "often the difference between topically similar
and actually answers the question" (architecture.md §2.4). Standard 2026 recipe:
retrieve top-20, rerank to top-5.

Model selection is env-driven via RERANKER_MODEL in config.py / .env.
Default: cross-encoder/ms-marco-MiniLM-L-6-v2 (22M params, lightweight for 512MB RAM hosting).
Alternative (local/unconstrained): BAAI/bge-reranker-v2-m3 (560M params).
"""

from __future__ import annotations

import logging
import math
from typing import Any

from src.config import (
    CONFIDENCE_THRESHOLD,
    RERANK_CANDIDATE_POOL,
    RERANK_FINAL_K,
    RERANKER_MODEL,
)
from src.retrieval.vector_search import SearchResult

logger = logging.getLogger(__name__)

# Module-level singleton
_reranker: Any = None


def _get_device() -> str:
    """Detect the best device for running the cross-encoder model."""
    import os
    if os.environ.get("FORCE_CPU_RERANKER") == "1":
        return "cpu"
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


class ONNXCrossEncoder:
    """Lightweight ONNX wrapper for cross-encoder/ms-marco-MiniLM-L-6-v2 (~25MB RSS vs ~300MB PyTorch RSS)."""

    def __init__(self, model_name: str = RERANKER_MODEL):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        from huggingface_hub import hf_hub_download

        self.model_name = model_name
        logger.info("Loading ONNX reranker tokenizer: %s", model_name)
        try:
            tok_json = hf_hub_download(repo_id=model_name, filename="tokenizer.json")
            self.tokenizer = Tokenizer.from_file(tok_json)
        except Exception:
            self.tokenizer = Tokenizer.from_pretrained(model_name)

        self.tokenizer.enable_padding(direction="right", pad_id=0, pad_token="[PAD]")
        self.tokenizer.enable_truncation(max_length=256)

        # Single-threaded SessionOptions to prevent thread pool allocation overhead on 512MB RAM ceiling
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        # Use pre-downloaded INT8 quantized ONNX model weights (~10MB Session RSS vs ~150MB FP32)
        try:
            onnx_path = hf_hub_download(repo_id="xenova/ms-marco-MiniLM-L-6-v2", filename="onnx/model_quantized.onnx")
        except Exception:
            onnx_path = hf_hub_download(repo_id="xenova/ms-marco-MiniLM-L-6-v2", filename="onnx/model.onnx")
        logger.info("Initializing ONNX InferenceSession for reranker (%s)", onnx_path)
        self.session = ort.InferenceSession(onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])

    def predict(self, pairs: list[list[str]], batch_size: int = 1) -> list[float]:
        if not pairs:
            return []
        import numpy as np

        all_logits: list[float] = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            encoded = self.tokenizer.encode_batch([[p[0], p[1]] for p in batch])
            input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
            attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
            token_type_ids = np.array([e.type_ids for e in encoded], dtype=np.int64)

            onnx_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
            outputs = self.session.run(None, onnx_inputs)
            logits = outputs[0]
            if len(logits.shape) > 1 and logits.shape[1] == 1:
                logits = logits.flatten()
            all_logits.extend([float(l) for l in logits])
        return all_logits



def _get_reranker():
    """Lazy-load the cross-encoder reranker model (ONNX preferred for low RAM footprint)."""
    global _reranker
    if _reranker is None:
        model_name = RERANKER_MODEL
        logger.info("Loading reranker model: %s", model_name)

        # Primary low-RAM path: ONNXCrossEncoder
        try:
            _reranker = ONNXCrossEncoder(model_name)
            logger.info("ONNXCrossEncoder loaded successfully (%s)", model_name)
            return _reranker
        except Exception as exc:
            logger.warning("Could not load ONNX reranker (%s), trying PyTorch CrossEncoder: %s", model_name, exc)

        device = _get_device()

        # If BGE model is specified and FlagEmbedding is available, try FlagReranker first
        if "bge-reranker" in model_name.lower():
            try:
                from FlagEmbedding import FlagReranker

                logger.info("Initializing FlagReranker on device: %s", device)
                _reranker = FlagReranker(model_name, devices=device, use_fp16=False)
                if device != "cpu":
                    _reranker.model.to(device)
                logger.info("FlagReranker loaded successfully (%s)", model_name)
                return _reranker
            except Exception as exc:
                logger.info("FlagReranker not used (%s), falling back to CrossEncoder", exc)

        # Fallback path: sentence-transformers CrossEncoder
        try:
            from sentence_transformers import CrossEncoder

            logger.info("Initializing CrossEncoder(%s) on device: %s", model_name, device)
            _reranker = CrossEncoder(model_name, device=device)
            logger.info("CrossEncoder loaded successfully (%s)", model_name)
        except Exception as exc:
            logger.warning("Failed to load reranker %s: %s — falling back", model_name, exc)
            _reranker = "unavailable"

    return _reranker


def _sigmoid(x: float) -> float:
    """Compute sigmoid function to map logit to [0, 1]."""
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def rerank(
    query: str,
    candidates: list[SearchResult],
    final_k: int | None = None,
    confidence_threshold: float | None = None,
    candidate_pool: int | None = None,
) -> tuple[list[SearchResult], bool]:
    """
    Rerank candidates using the cross-encoder, return top-k.

    Implements the "no context found" check (FR-10, design.md §4):
    if the top reranked chunk's score is below the confidence threshold,
    context_found is set to False.

    Args:
        query: The user's natural-language question.
        candidates: Pre-fused candidate SearchResults.
        final_k: Number of results to keep after reranking.
        confidence_threshold: Score below which we declare "no context found."
        candidate_pool: Max number of candidates passed into cross-encoder.

    Returns:
        (top_k_results, context_found) — context_found is False if the best
        result is below the confidence threshold.
    """
    final_k = final_k or RERANK_FINAL_K
    confidence_threshold = confidence_threshold or CONFIDENCE_THRESHOLD
    candidate_pool = candidate_pool or RERANK_CANDIDATE_POOL

    if not candidates:
        return [], False

    # Cap candidates to candidate_pool to prevent latency blowup
    if len(candidates) > candidate_pool:
        logger.debug(
            "Capping reranker candidates from %d to top %d", len(candidates), candidate_pool
        )
        candidates = candidates[:candidate_pool]

    reranker = _get_reranker()

    # Fallback if reranker unavailable: return top-k by existing scores
    if reranker == "unavailable":
        logger.warning("Reranker unavailable, using pre-fusion scores")
        top = sorted(candidates, key=lambda r: r.score, reverse=True)[:final_k]
        context_found = len(top) > 0 and top[0].score > 0
        return top, context_found

    pairs = [[query, c.text] for c in candidates]

    try:
        if hasattr(reranker, "compute_score"):
            # FlagReranker API
            raw_scores = reranker.compute_score(pairs, normalize=True)
            if isinstance(raw_scores, (int, float)):
                raw_scores = [raw_scores]
            scores = [float(s) for s in raw_scores]
        elif hasattr(reranker, "predict"):
            # CrossEncoder API (batch_size=1 keeps memory footprint minimal under 512MB limit)
            import ctypes
            import gc

            raw_scores = reranker.predict(pairs, batch_size=1)
            if hasattr(raw_scores, "tolist"):
                raw_scores = raw_scores.tolist()
            if isinstance(raw_scores, (int, float)):
                raw_scores = [raw_scores]
            # Map raw logits to [0, 1] via sigmoid if not already normalized
            scores = [_sigmoid(float(s)) for s in raw_scores]

            # Return freed intermediate tensors to Linux kernel immediately
            gc.collect()
            try:
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception:
                pass
        else:
            raise ValueError(f"Unknown reranker model interface: {type(reranker)}")
    except Exception as exc:
        logger.error("Reranker scoring failed: %s — falling back", exc)
        top = sorted(candidates, key=lambda r: r.score, reverse=True)[:final_k]
        context_found = len(top) > 0 and top[0].score > 0
        return top, context_found

    # Attach reranker scores to candidates
    scored = list(zip(candidates, scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    top_results: list[SearchResult] = []
    for candidate, score in scored[:final_k]:
        top_results.append(
            SearchResult(
                chunk_id=candidate.chunk_id,
                text=candidate.text,
                score=float(score),
                source_url=candidate.source_url,
                section_heading=candidate.section_heading,
                doc_category=candidate.doc_category,
            )
        )

    # "No context found" check — FR-10
    context_found = len(top_results) > 0 and top_results[0].score >= confidence_threshold

    logger.debug(
        "Reranked %d → %d candidates (top score: %.3f, threshold: %.3f, context_found: %s)",
        len(candidates),
        len(top_results),
        top_results[0].score if top_results else 0,
        confidence_threshold,
        context_found,
    )
    return top_results, context_found

