"""
_DEPRECATED — This Streamlit frontend has been superseded by the Next.js app in
`frontend-web/`. It is kept here as a record of the earlier implementation.
Do not run this in production — use `frontend-web/` instead.

Original module docstring follows:
---------------------------------------------------------------------------
Streamlit frontend for the Colophon RAG Knowledge Assistant.

Why Streamlit: design.md §8, architecture.md §2.8 — ships a clean, functional UI
quickly to preserve engineering budget for retrieval quality and evaluation.
Displays answer text, expandable cited sources with links and relevance scores,
latency, and backend metadata with required portfolio disclaimer.
"""

from __future__ import annotations

import os
import requests
import streamlit as st

API_HOST = os.getenv("API_HOST", "http://localhost:8000")

st.set_page_config(
    page_title="Colophon — Claude API Assistant",
    page_icon="📜",
    layout="centered",
)

st.title("📜 Colophon — Claude API Knowledge Assistant")
st.caption(
    "Grounded Q&A assistant powered by hybrid retrieval (dense vector + BM25 keyword) "
    "and cross-encoder reranking over official Anthropic Claude API documentation."
)

# Sidebar with system info & settings
with st.sidebar:
    st.header("⚙️ Configuration")
    api_url = st.text_input("FastAPI Endpoint", value=API_HOST)
    top_k = st.slider("Top Sources to Cite", min_value=1, max_value=10, value=5)

    st.markdown("---")
    st.markdown("### 🔍 System Architecture")
    st.markdown(
        """
        - **Corpus:** Anthropic Claude API Docs
        - **Retrieval:** Hybrid (ChromaDB + BM25 Okapi)
        - **Fusion:** Reciprocal Rank Fusion (RRF)
        - **Reranker:** BGE-reranker-v2 (Cross-Encoder)
        - **Embeddings:** all-MiniLM-L6-v2
        - **Grounding:** Strict prompt constraint (Zero-hallucination refusal on OOD)
        """
    )
    st.markdown("---")
    try:
        health_resp = requests.get(f"{api_url}/health", timeout=3)
        if health_resp.status_code == 200:
            health_data = health_resp.json()
            st.success(f"API Online | Backend: `{health_data.get('backend')}`")
            st.info(f"Chunks Indexed: **{health_data.get('chunks_indexed', 0)}**")
        else:
            st.warning("API reachable but returned non-200")
    except Exception:
        st.error("API is offline or unreachable")

# Main Query Interface
question = st.text_area(
    "Ask a question about the Anthropic Claude API:",
    placeholder="e.g. How do I force Claude to use a specific tool in the Messages API?",
    height=100,
)

col1, col2 = st.columns([1, 4])
with col1:
    ask_button = st.button("Submit Question", type="primary", width="stretch")

if ask_button and question.strip():
    with st.spinner("Searching documentation and generating grounded answer..."):
        try:
            payload = {
                "question": question.strip(),
                "top_k": top_k,
            }
            resp = requests.post(f"{api_url}/query", json=payload, timeout=60)

            if resp.status_code == 200:
                data = resp.json()
                answer = data.get("answer", "")
                sources = data.get("sources", [])
                latency_ms = data.get("latency_ms", 0)
                backend_used = data.get("backend_used", "unknown")
                context_found = data.get("context_found", True)

                # Render Answer
                st.subheader("Answer")
                if not context_found:
                    st.warning(answer)
                else:
                    st.markdown(answer)

                # Render Sources
                if sources:
                    st.markdown("---")
                    st.subheader(f"📚 Grounded Sources ({len(sources)})")
                    for idx, src in enumerate(sources, 1):
                        heading = src.get("section_heading") or "General Section"
                        url = src.get("source_url", "")
                        score = src.get("relevance_score", 0.0)
                        text = src.get("text", "")

                        with st.expander(f"[{idx}] {heading} (Score: {score:.4f})"):
                            if url:
                                st.markdown(f"🔗 **URL:** [{url}]({url})")
                            st.code(text, language="markdown")

                # Metrics Footer
                st.markdown("---")
                cols = st.columns(3)
                cols[0].metric("Latency", f"{latency_ms} ms")
                cols[1].metric("Inference Backend", backend_used.capitalize())
                cols[2].metric("Context Matched", "Yes" if context_found else "No")

            else:
                st.error(f"API Error ({resp.status_code}): {resp.text}")

        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the FastAPI backend. Please make sure `uvicorn src.api.main:app` is running.")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

# Mandatory Disclaimer (design.md §8, rules.md §2)
st.markdown("---")
st.caption(
    "⚠️ **Disclaimer:** Colophon is an independent portfolio project answering questions from "
    "Anthropic's public Claude API documentation as of August 31, 2026. "
    "Not affiliated with or endorsed by Anthropic."
)
