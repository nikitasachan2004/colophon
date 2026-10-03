/**
 * Static data derived from SRS/memory.md and the canonical Phase 4 ablation results.
 * Single source of truth for all numbers displayed in the UI — change here, updates everywhere.
 *
 * RAGAS scores from: phase4_plus_reranking_final___production_20260903_235247.json
 * Judge: llama3.1:8b via Ollama (local, zero quota). All 4 stages: 0 generation errors.
 * Latency from: latency_optionB_benchmark.log (CPU reranker, /query endpoint, 20 questions)
 */

export const PROJECT_STATS = {
  chunksIndexed: 2621,
  pagesIndexed: 77,
  evalQuestions: 25,
  recallAtFive: 0.90,
  faithfulness: 0.706,
  contextPrecision: 0.861,
  answerRelevancy: 0.746,
  contextRecall: 0.937,
  oodAccuracy: 1.00,
  latencyP50Ms: 7618,
  latencyP95Ms: 11013,
  testsPass: 20,
  costToBuild: "$0",
} as const;

export interface AblationRow {
  stage: string;
  retrieval: string;
  reranking: string;
  recallAt5: number;
  oodAcc: number;
  faithfulness: number;
  ansRelevancy: number;
  ctxPrecision: number;
  ctxRecall: number;
  avgLatencyMs: number;
  resultFile: string;
}

export const ABLATION_RESULTS: AblationRow[] = [
  {
    stage: "Baseline (naive chunks)",
    retrieval: "Vector-only",
    reranking: "None",
    recallAt5: 0.800,
    oodAcc: 1.000,
    faithfulness: 0.608,
    ansRelevancy: 0.677,
    ctxPrecision: 0.748,
    ctxRecall: 0.997,
    avgLatencyMs: 1144,
    resultFile: "phase4_baseline_naive_chunks,_vector-only_20260902_204914.json",
  },
  {
    stage: "+ Structure-Aware Chunking",
    retrieval: "Vector-only",
    reranking: "None",
    recallAt5: 0.800,
    oodAcc: 1.000,
    faithfulness: 0.685,
    ansRelevancy: 0.651,
    ctxPrecision: 0.748,
    ctxRecall: 0.997,
    avgLatencyMs: 1475,
    resultFile: "phase4_plus_structure-aware_chunking_20260903_140340.json",
  },
  {
    stage: "+ Hybrid Retrieval (V+BM25+RRF)",
    retrieval: "Hybrid (Dense + BM25 + RRF)",
    reranking: "None",
    recallAt5: 0.900,
    oodAcc: 1.000,
    faithfulness: 0.564,
    ansRelevancy: 0.748,
    ctxPrecision: 0.799,
    ctxRecall: 0.950,
    avgLatencyMs: 1602,
    resultFile: "phase4_plus_hybrid_retrieval_vector_plus_bm25_plus_rrf_20260903_180606.json",
  },
  {
    stage: "+ Reranking (Final / Production)",
    retrieval: "Hybrid (Dense + BM25 + RRF)",
    reranking: "BGE-reranker-v2-m3",
    recallAt5: 0.900,
    oodAcc: 1.000,
    faithfulness: 0.706,
    ansRelevancy: 0.746,
    ctxPrecision: 0.861,
    ctxRecall: 0.937,
    avgLatencyMs: 33308,
    resultFile: "phase4_plus_reranking_final___production_20260903_235247.json",
  },
];

export const TECH_STACK = [
  { layer: "Scraping", choice: "requests + BeautifulSoup", cost: "Free" },
  { layer: "Chunking", choice: "Custom structure-aware splitter (heading-based + overlap)", cost: "Free" },
  { layer: "Embeddings", choice: "sentence-transformers/all-MiniLM-L6-v2", cost: "Free, local" },
  { layer: "Vector store", choice: "ChromaDB (persisted local)", cost: "Free" },
  { layer: "Sparse retrieval", choice: "rank_bm25 (BM25Okapi)", cost: "Free" },
  { layer: "Reranker", choice: "BAAI/bge-reranker-v2-m3 (cross-encoder)", cost: "Free, local" },
  { layer: "Fusion", choice: "Reciprocal Rank Fusion (RRF, k=60)", cost: "Free" },
  { layer: "LLM (dev)", choice: "Ollama — llama3.1:8b", cost: "Free, local" },
  { layer: "LLM (prod)", choice: "Groq — openai/gpt-oss-120b", cost: "Free tier" },
  { layer: "Backend", choice: "FastAPI + uvicorn", cost: "Free" },
  { layer: "Frontend", choice: "Next.js 16 (App Router) + Tailwind + shadcn", cost: "Free" },
  { layer: "Evaluation", choice: "RAGAS 0.3.9 + Ollama judge (llama3.1:8b)", cost: "Free" },
  { layer: "Backend hosting", choice: "Render (Docker, free tier)", cost: "Free" },
  { layer: "Frontend hosting", choice: "Vercel (free tier)", cost: "Free" },
  { layer: "CI", choice: "GitHub Actions", cost: "Free (public repo)" },
] as const;

// Abstract gradient images from Unsplash (free to use, no attribution required under Unsplash license)
// These replace the stock photos in the original ImageStreamHero demo.
export const HERO_IMAGES = [
  "https://images.unsplash.com/photo-1518770660439-4636190af475?w=1200&auto=format&fit=crop&q=80",
  "https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=1200&auto=format&fit=crop&q=80",
  "https://images.unsplash.com/photo-1531297484001-80022131f5a1?w=1200&auto=format&fit=crop&q=80",
  "https://images.unsplash.com/photo-1518770660439-4636190af475?w=1200&auto=format&fit=crop&q=80",
  "https://images.unsplash.com/photo-1451187580459-43490279c0fa?w=1200&auto=format&fit=crop&q=80",
  "https://images.unsplash.com/photo-1504639725590-34d0984388bd?w=1200&auto=format&fit=crop&q=80",
  "https://images.unsplash.com/photo-1526374965328-7f61d4dc18c5?w=1200&auto=format&fit=crop&q=80",
  "https://images.unsplash.com/photo-1620641788421-7a1c342ea42e?w=1200&auto=format&fit=crop&q=80",
] as const;
