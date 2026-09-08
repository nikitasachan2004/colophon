"use client";

import { motion } from "motion/react";
import { Database, Search, GitMerge, ListFilter, Play, Server, Layers, ShieldCheck } from "lucide-react";

export default function ArchitecturePage() {
  const containerVariants = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: { staggerChildren: 0.1 }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0 }
  };

  return (
    <div className="min-h-screen bg-canvas text-ink pt-32 pb-24 selection:bg-white/20">
      {/* Hero Header */}
      <section className="w-full px-8 pb-16 border-b border-hairline">
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="max-w-4xl mx-auto text-center space-y-6"
        >
          <h1 className="font-display text-5xl sm:text-6xl font-bold tracking-tight leading-tight bg-gradient-to-br from-white to-white/50 bg-clip-text text-transparent">
            System Architecture
          </h1>
          <p className="font-text text-lg text-muted leading-relaxed max-w-2xl mx-auto">
            Production retrieval-augmented generation: hybrid search, reranking, grounded generation.
          </p>
        </motion.div>
      </section>

      {/* Main Content */}
      <section className="w-full py-16 px-8 overflow-hidden">
        <div className="max-w-6xl mx-auto space-y-24">
          {/* Ingestion Pipeline */}
          <motion.div 
            variants={containerVariants}
            initial="hidden"
            whileInView="show"
            viewport={{ once: true, margin: "-100px" }}
            className="space-y-8"
          >
            <div className="space-y-2">
              <h2 className="font-display text-3xl font-semibold tracking-tight flex items-center gap-3">
                <Database className="w-8 h-8 text-blue-500" />
                I. Ingestion (Offline)
              </h2>
              <p className="font-text text-lg text-muted">
                Enterprise documentation is scraped, structured, and indexed once. All computation happens in batch.
              </p>
            </div>

            <div className="grid md:grid-cols-5 gap-4 relative">
              <div className="absolute top-1/2 left-0 right-0 h-0.5 bg-gradient-to-r from-blue-500/0 via-blue-500/20 to-purple-500/0 -z-10 hidden md:block" />
              {[
                { num: "01", title: "Scraper", desc: "Fetch enterprise docs" },
                { num: "02", title: "Chunker", desc: "Split by headings, 12% overlap" },
                { num: "03", title: "Embedder", desc: "all-MiniLM-L6-v2 vectors" },
                { num: "04", title: "Vector DB", desc: "ChromaDB, 2,621 chunks" },
                { num: "05", title: "BM25 Index", desc: "Keyword search fallback" },
              ].map((stage, i) => (
                <motion.div key={stage.num} variants={itemVariants} className="glass-card p-6 relative group overflow-hidden">
                  <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 to-purple-500/5 opacity-0 group-hover:opacity-100 transition-opacity" />
                  <div className="font-mono text-xs text-blue-400/80 mb-3 bg-blue-500/10 w-fit px-2 py-1 rounded-md">
                    {stage.num}
                  </div>
                  <h3 className="font-display font-medium text-base mb-2 group-hover:text-blue-400 transition-colors">
                    {stage.title}
                  </h3>
                  <p className="font-text text-sm text-muted">
                    {stage.desc}
                  </p>
                </motion.div>
              ))}
            </div>
          </motion.div>

          {/* Query-Time Pipeline */}
          <motion.div 
            variants={containerVariants}
            initial="hidden"
            whileInView="show"
            viewport={{ once: true, margin: "-100px" }}
            className="space-y-8 pt-12 border-t border-hairline relative"
          >
            <div className="absolute left-1/2 -top-px w-32 h-px bg-gradient-to-r from-transparent via-purple-500/50 to-transparent -translate-x-1/2" />
            
            <div className="space-y-2">
              <h2 className="font-display text-3xl font-semibold tracking-tight flex items-center gap-3">
                <Play className="w-8 h-8 text-purple-500" />
                II. Query Pipeline (Online)
              </h2>
              <p className="font-text text-lg text-muted">
                User query flows through dual-index search, fusion, reranking, and LLM generation.
              </p>
            </div>

            <div className="space-y-12">
              {/* Stage 1: Dual Search */}
              <motion.div variants={itemVariants} className="relative pl-8 md:pl-0">
                <div className="hidden md:flex absolute -left-12 w-8 h-8 rounded-full glass-card items-center justify-center text-sm font-mono text-purple-400">1</div>
                <div className="md:ml-0 md:pl-12 border-l border-hairline md:border-none space-y-6">
                  <div className="flex items-center gap-3">
                    <Search className="w-5 h-5 text-muted" />
                    <h3 className="font-display text-xl font-medium tracking-tight">
                      Dual-Index Search
                    </h3>
                  </div>
                  <div className="grid md:grid-cols-2 gap-6">
                    <div className="glass-card p-6">
                      <h4 className="font-mono text-xs uppercase tracking-wider text-muted mb-3 flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-blue-500" /> Dense Search
                      </h4>
                      <p className="font-text text-body">
                        Cosine similarity in ChromaDB vector space. Top-20 by semantic relevance.
                      </p>
                    </div>
                    <div className="glass-card p-6">
                      <h4 className="font-mono text-xs uppercase tracking-wider text-muted mb-3 flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-amber-500" /> Sparse Search
                      </h4>
                      <p className="font-text text-body">
                        BM25 keyword matching. Top-20 by exact term frequency.
                      </p>
                    </div>
                  </div>
                </div>
              </motion.div>

              {/* Stage 2: Fusion */}
              <motion.div variants={itemVariants} className="relative pl-8 md:pl-0">
                <div className="hidden md:flex absolute -left-12 w-8 h-8 rounded-full glass-card items-center justify-center text-sm font-mono text-purple-400">2</div>
                <div className="md:ml-0 md:pl-12 border-l border-hairline md:border-none space-y-4">
                  <div className="flex items-center gap-3">
                    <GitMerge className="w-5 h-5 text-muted" />
                    <h3 className="font-display text-xl font-medium tracking-tight">
                      Reciprocal Rank Fusion
                    </h3>
                  </div>
                  <p className="font-text text-body text-lg">
                    Combine dense and sparse results using RRF (k=60) to reduce redundancy and leverage both signals.
                  </p>
                  <div className="glass-card p-4 font-mono text-sm text-blue-400/80 bg-blue-500/5 inline-block">
                    score = Σ(1 / (k + rank)) for each index
                  </div>
                </div>
              </motion.div>

              {/* Stage 3: Reranking */}
              <motion.div variants={itemVariants} className="relative pl-8 md:pl-0">
                <div className="hidden md:flex absolute -left-12 w-8 h-8 rounded-full glass-card items-center justify-center text-sm font-mono text-purple-400">3</div>
                <div className="md:ml-0 md:pl-12 border-l border-hairline md:border-none space-y-6">
                  <div className="flex items-center gap-3">
                    <ListFilter className="w-5 h-5 text-muted" />
                    <h3 className="font-display text-xl font-medium tracking-tight">
                      Cross-Encoder Reranking
                    </h3>
                  </div>
                  <p className="font-text text-body text-lg">
                    BGE-reranker-v2-m3 scores question-chunk pairs. Top-5 passed to LLM.
                  </p>
                  <div className="grid md:grid-cols-3 gap-4">
                    {[
                      { l: "Input", v: "Fused top-20 + query" },
                      { l: "Model", v: "BGE cross-encoder 768d" },
                      { l: "Output", v: "Top-5 by relevance" }
                    ].map(x => (
                      <div key={x.l} className="glass-card p-5 border-white/5">
                        <div className="font-mono text-[10px] uppercase tracking-wider text-muted mb-2">{x.l}</div>
                        <p className="font-text text-sm text-ink">{x.v}</p>
                      </div>
                    ))}
                  </div>
                </div>
              </motion.div>

              {/* Stage 4: Confidence Filtering */}
              <motion.div variants={itemVariants} className="relative pl-8 md:pl-0">
                <div className="hidden md:flex absolute -left-12 w-8 h-8 rounded-full glass-card items-center justify-center text-sm font-mono text-purple-400">4</div>
                <div className="md:ml-0 md:pl-12 border-l border-hairline md:border-none space-y-2">
                  <div className="flex items-center gap-3">
                    <ShieldCheck className="w-5 h-5 text-muted" />
                    <h3 className="font-display text-xl font-medium tracking-tight">
                      Confidence Threshold
                    </h3>
                  </div>
                  <p className="font-text text-body text-lg">
                    If max reranker score is below threshold, return explicit refusal instead of hallucinating.
                  </p>
                </div>
              </motion.div>

              {/* Stage 5: Generation */}
              <motion.div variants={itemVariants} className="relative pl-8 md:pl-0">
                <div className="hidden md:flex absolute -left-12 w-8 h-8 rounded-full glass-card items-center justify-center text-sm font-mono text-purple-400">5</div>
                <div className="md:ml-0 md:pl-12 border-l border-hairline md:border-none space-y-6">
                  <div className="flex items-center gap-3">
                    <Layers className="w-5 h-5 text-muted" />
                    <h3 className="font-display text-xl font-medium tracking-tight">
                      LLM Generation
                    </h3>
                  </div>
                  <p className="font-text text-body text-lg">
                    Model generates answer strictly from top-5 chunks with inline citations.
                  </p>
                  <div className="grid md:grid-cols-2 gap-4">
                    <div className="glass-card p-6 border-white/5 bg-gradient-to-br from-white/5 to-transparent">
                      <h4 className="font-mono text-[10px] uppercase tracking-wider text-muted mb-2">Development</h4>
                      <p className="font-text text-ink">Ollama llama3.1:8b (local, free)</p>
                    </div>
                    <div className="glass-card p-6 border-blue-500/20 bg-gradient-to-br from-blue-500/10 to-transparent">
                      <h4 className="font-mono text-[10px] uppercase tracking-wider text-blue-400 mb-2">Production</h4>
                      <p className="font-text text-ink">Groq (cloud API, fast inference)</p>
                    </div>
                  </div>
                </div>
              </motion.div>
            </div>
          </motion.div>

          {/* Tech Stack */}
          <motion.div 
            variants={containerVariants}
            initial="hidden"
            whileInView="show"
            viewport={{ once: true, margin: "-100px" }}
            className="space-y-8 pt-12 border-t border-hairline relative"
          >
            <div className="absolute left-1/2 -top-px w-32 h-px bg-gradient-to-r from-transparent via-blue-500/50 to-transparent -translate-x-1/2" />
            <h2 className="font-display text-3xl font-semibold tracking-tight flex items-center gap-3">
              <Server className="w-8 h-8 text-blue-500" />
              Technology Stack
            </h2>
            <div className="grid md:grid-cols-3 gap-6">
              {[
                { category: "Retrieval", items: ["ChromaDB (vectors)", "rank_bm25 (keywords)", "RRF (fusion)"] },
                { category: "Reranking", items: ["BGE-reranker-v2-m3", "Hugging Face", "Cross-encoder"] },
                { category: "Generation", items: ["Groq API (prod)", "Ollama (dev)", "Fast inference"] },
                { category: "Backend", items: ["FastAPI", "Python", "Async/await"] },
                { category: "Frontend", items: ["Next.js 16", "React 19", "TypeScript"] },
                { category: "Deployment", items: ["Vercel (UI)", "Render (API)", "Free tiers"] },
              ].map((section) => (
                <motion.div key={section.category} variants={itemVariants} className="glass-card p-6 hover:-translate-y-1 transition-transform">
                  <h3 className="font-display font-medium text-base mb-4 text-ink">
                    {section.category}
                  </h3>
                  <ul className="space-y-3">
                    {section.items.map((item) => (
                      <li key={item} className="font-text text-sm text-muted flex items-center gap-2">
                        <span className="w-1 h-1 rounded-full bg-blue-500/50" />
                        {item}
                      </li>
                    ))}
                  </ul>
                </motion.div>
              ))}
            </div>
          </motion.div>

          {/* Key Properties */}
          <motion.div 
            variants={containerVariants}
            initial="hidden"
            whileInView="show"
            viewport={{ once: true, margin: "-100px" }}
            className="space-y-8 pt-12 border-t border-hairline relative"
          >
            <h2 className="font-display text-3xl font-semibold tracking-tight">
              Key Properties
            </h2>
            <div className="grid md:grid-cols-2 gap-6">
              {[
                { title: "Zero Hallucinations", desc: "Explicit refusal when confidence is low. No made-up answers." },
                { title: "Real-time Evaluation", desc: "RAGAS scoring on every query. Faithfulness, recall, precision tracked." },
                { title: "Production Ready", desc: "Deployed on Vercel + Groq. 100% free tier compatible." },
                { title: "Grounded Citations", desc: "Every answer inline-cited to source documentation with URLs." },
              ].map((prop) => (
                <motion.div key={prop.title} variants={itemVariants} className="glass-card p-8 group">
                  <h3 className="font-display font-medium text-lg mb-2 text-ink group-hover:text-purple-400 transition-colors">
                    {prop.title}
                  </h3>
                  <p className="font-text text-body">
                    {prop.desc}
                  </p>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </div>
      </section>
    </div>
  );
}
