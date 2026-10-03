"use client"

import Link from "next/link";
import { PrismHero } from "@/components/ui/prism-hero";
import { motion } from "motion/react";
import { PROJECT_STATS } from "@/lib/data";

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-canvas text-ink selection:bg-white/20">
      {/* Hero Section using PrismHero */}
      <PrismHero
        topInset
        eyebrow="Generation → Retrieval → Grounding"
        headline="Grounded"
        description="A production retrieval-augmented generation system for enterprise documentation. Every answer grounded in real sources. No hallucinations."
        meta={["Hybrid Search", "Cross-Encoder Reranking", "Zero Hallucinations"]}
        action={
          <Link href="/ask" className="btn-primary">
            Ask A Question
          </Link>
        }
        secondaryAction={
          <Link
            href="/evaluation"
            className="inline-flex items-center justify-center h-12 px-8 rounded-full border border-white/20 font-medium text-sm text-white/80 transition-all hover:bg-white/10 hover:border-white/50 backdrop-blur-md"
          >
            View Results
          </Link>
        }
        background="#09090b"
        foreground="#ffffff"
        accent="#3b82f6"
        dispersion={0.8}
        tint="#e0f2fe"
        displayFont="var(--font-sans)"
      />

      {/* Divider */}
      <div className="w-full h-px bg-gradient-to-r from-transparent via-white/10 to-transparent" />

      {/* Benchmarks Section - Glassmorphic */}
      <section className="relative w-full py-32 px-8 overflow-hidden">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-blue-900/20 via-canvas to-canvas -z-10" />
        <div className="max-w-6xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-sm font-mono tracking-widest text-accent-blue uppercase mb-4">Measured</h2>
            <p className="text-xl text-muted max-w-2xl mx-auto">
              Not estimated. Every metric on a held-out benchmark, evaluated with RAGAS.
            </p>
          </motion.div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            {[
              { label: "Chunks", value: PROJECT_STATS.chunksIndexed.toLocaleString(), caption: `${PROJECT_STATS.pagesIndexed} pages indexed` },
              { label: "Faithfulness", value: PROJECT_STATS.faithfulness.toString(), caption: "RAGAS score" },
              { label: "Context", value: PROJECT_STATS.contextPrecision.toString(), caption: "Precision" },
              { label: "Cost", value: PROJECT_STATS.costToBuild, caption: "Free tier" },
            ].map((stat, i) => (
              <motion.div
                key={stat.label}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.1 }}
                className="glass-card p-8 flex flex-col items-center justify-center text-center group"
              >
                <div className="text-4xl sm:text-5xl font-display font-bold text-white mb-2 group-hover:scale-110 transition-transform duration-500">
                  {stat.value}
                </div>
                <div className="text-sm font-semibold tracking-wider text-accent-blue uppercase mb-1">
                  {stat.label}
                </div>
                <div className="text-xs text-muted-soft">
                  {stat.caption}
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      <div className="w-full h-px bg-gradient-to-r from-transparent via-white/10 to-transparent" />

      {/* The Pipeline Section */}
      <section className="w-full py-32 px-8 relative">
        <div className="max-w-4xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-sm font-mono tracking-widest text-accent-purple uppercase mb-4">The Pipeline</h2>
            <p className="text-xl text-muted max-w-2xl mx-auto">
              Four stages from question to grounded answer.
            </p>
          </motion.div>

          <div className="space-y-6">
            {[
              {
                number: "01",
                title: "Ingestion",
                description: "Enterprise documentation is scraped, chunked along heading hierarchies, and embedded.",
              },
              {
                number: "02",
                title: "Hybrid Retrieval",
                description: "Your question searches both dense vector and sparse BM25 indexes simultaneously, fused using RRF.",
              },
              {
                number: "03",
                title: "Reranking",
                description: "BGE-reranker cross-encoder re-scores results. A confidence threshold filters low-signal candidates.",
              },
              {
                number: "04",
                title: "Generation",
                description: "Generates strictly from the top-5 chunks. An explicit refusal triggers if confidence is below threshold.",
              },
            ].map((stage, i) => (
              <motion.div
                key={stage.number}
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.1 }}
                className="glass-card p-6 sm:p-8 flex flex-col sm:flex-row gap-6 items-start"
              >
                <div className="shrink-0 text-3xl font-mono font-bold text-white/20">
                  {stage.number}
                </div>
                <div>
                  <h3 className="text-xl font-display font-semibold text-white mb-2">
                    {stage.title}
                  </h3>
                  <p className="text-muted leading-relaxed">
                    {stage.description}
                  </p>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      <div className="w-full h-px bg-gradient-to-r from-transparent via-white/10 to-transparent" />

      {/* Capabilities Section */}
      <section className="w-full py-32 px-8 relative overflow-hidden">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_bottom,_var(--tw-gradient-stops))] from-blue-900/10 via-canvas to-canvas -z-10" />
        <div className="max-w-6xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-sm font-mono tracking-widest text-blue-400 uppercase mb-4">Capabilities</h2>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {[
              { title: "Hybrid Search", desc: "Dense vectors + BM25, fused with RRF" },
              { title: "Reranking", desc: "BGE cross-encoder for answer relevance" },
              { title: "Confidence Filtering", desc: "Threshold-based refusal on low signals" },
              { title: "RAGAS Evaluation", desc: "Real benchmarks on held-out questions" },
              { title: "Production Ready", desc: "Vercel + Groq deployment, free tiers" },
              { title: "Open Source", desc: "Full codebase, single-click deployment" },
            ].map((feature, i) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, scale: 0.95 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.1 }}
                className="glass-card p-8 group hover:-translate-y-1"
              >
                <h3 className="text-lg font-display font-semibold text-white mb-2 group-hover:text-blue-400 transition-colors">
                  {feature.title}
                </h3>
                <p className="text-sm text-muted">
                  {feature.desc}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Final CTA */}
      <section className="w-full py-32 px-8 relative">
        <div className="max-w-3xl mx-auto text-center space-y-8 glass-card p-12 sm:p-16">
          <h2 className="text-4xl sm:text-5xl font-display font-bold text-white leading-tight text-gradient">
            Ready to ask?
          </h2>
          <p className="text-lg text-muted">
            Ask any question about your enterprise knowledge base. Your answer will be grounded in official documentation with inline citations.
          </p>
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 pt-4">
            <Link href="/ask" className="btn-primary">
              Ask a Question
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
}
