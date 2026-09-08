"use client";

import * as React from "react";
import { Loader2, ExternalLink, Search, ArrowRight } from "lucide-react";
import { motion } from "motion/react";

interface SourceChunk {
  text: string;
  source_url: string;
  section_heading: string;
  relevance_score: number;
}

interface QueryResponse {
  answer: string;
  sources: SourceChunk[];
  latency_ms: number;
  backend_used: string;
  context_found: boolean;
}

export default function AskPage() {
  const [query, setQuery] = React.useState("");
  const [response, setResponse] = React.useState<QueryResponse | null>(null);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const [stage, setStage] = React.useState<"idle" | "retrieving" | "generating">("idle");

  const exampleQuestions = [
    "What is extended thinking in Claude?",
    "How do I use tool use with the Claude API?",
    "What are the rate limits for Claude models?",
    "How does prompt caching work and when should I use it?",
  ];

  const handleAsk = async (question: string) => {
    setQuery(question);
    setLoading(true);
    setError(null);
    setResponse(null);

    const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

    try {
      // Simulate retrieval stage
      setStage("retrieving");
      await new Promise((resolve) => setTimeout(resolve, 800));

      // Simulate generation stage
      setStage("generating");

      const res = await fetch(`${apiUrl}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });

      if (!res.ok) throw new Error(`API error: ${res.status}`);
      const data: QueryResponse = await res.json();
      setResponse(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to fetch answer");
    } finally {
      setLoading(false);
      setStage("idle");
    }
  };

  return (
    <div className="min-h-screen bg-canvas text-ink pt-32 pb-24 selection:bg-white/20">
      {/* Hero Header */}
      <section className="w-full px-8 pb-12">
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="max-w-3xl mx-auto text-center space-y-4"
        >
          <h1 className="font-display text-4xl sm:text-5xl font-bold tracking-tight leading-tight">
            Knowledge Assistant
          </h1>
          <p className="font-text text-lg text-muted leading-relaxed">
            Every answer grounded in official enterprise documentation.
          </p>
        </motion.div>
      </section>

      {/* Main Content */}
      <section className="w-full px-8">
        <div className="max-w-3xl mx-auto space-y-12">
          {/* Input Section */}
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="relative group"
          >
            <div className="absolute -inset-0.5 bg-gradient-to-r from-blue-500/30 to-purple-500/30 rounded-2xl blur opacity-30 group-hover:opacity-50 transition duration-500" />
            <div className="relative flex items-center bg-surface-soft border border-hairline rounded-2xl overflow-hidden focus-within:border-hairline-strong transition-colors p-2 shadow-2xl">
              <div className="pl-4 pr-2 text-muted">
                <Search className="w-5 h-5" />
              </div>
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !loading) handleAsk(query);
                }}
                placeholder="Ask anything about the system..."
                className="w-full bg-transparent text-ink placeholder-muted py-3 px-2 focus:outline-none text-lg"
              />
              <button
                onClick={() => handleAsk(query)}
                disabled={!query.trim() || loading}
                className="ml-2 flex items-center justify-center w-12 h-12 rounded-xl bg-white text-black transition-transform hover:scale-105 active:scale-95 disabled:opacity-50 disabled:hover:scale-100"
              >
                {loading ? (
                  <Loader2 className="w-5 h-5 animate-spin" />
                ) : (
                  <ArrowRight className="w-5 h-5" />
                )}
              </button>
            </div>
          </motion.div>

          {/* Example Questions */}
          {!response && !loading && !error && (
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.2 }}
              className="space-y-4"
            >
              <div className="grid sm:grid-cols-2 gap-4">
                {exampleQuestions.map((q, i) => (
                  <button
                    key={q}
                    onClick={() => handleAsk(q)}
                    disabled={loading}
                    className="text-left p-5 glass-card group hover:-translate-y-1 transition-all duration-300"
                  >
                    <p className="font-text text-sm text-body group-hover:text-ink transition-colors leading-relaxed">{q}</p>
                  </button>
                ))}
              </div>
            </motion.div>
          )}

          {/* Processing Stage Indicator */}
          {loading && (
            <motion.div 
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="py-16 text-center space-y-6"
            >
              <div className="relative w-16 h-16 mx-auto">
                <div className="absolute inset-0 rounded-full border-t-2 border-l-2 border-blue-400 animate-spin" />
                <div className="absolute inset-2 rounded-full border-r-2 border-b-2 border-purple-400 animate-spin direction-reverse" style={{ animationDirection: "reverse" }} />
              </div>
              <p className="font-mono text-xs uppercase tracking-widest text-muted animate-pulse">
                {stage === "retrieving" && "Searching documentation..."}
                {stage === "generating" && "Generating answer..."}
              </p>
            </motion.div>
          )}

          {/* Error State */}
          {error && (
            <motion.div 
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="p-6 border border-red-500/20 bg-red-500/5 rounded-2xl"
            >
              <p className="font-text text-sm text-red-400">
                {error}
              </p>
              <p className="font-text text-xs text-muted mt-2">
                The backend may be starting up — please try again in a moment.
              </p>
            </motion.div>
          )}

          {/* Answer Display */}
          {response && !loading && (
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="space-y-12"
            >
              {/* Answer Section */}
              <div className="glass-card p-8 sm:p-10">
                <div className="flex items-center justify-between mb-6 pb-6 border-b border-hairline">
                  <h2 className="font-display font-semibold text-xl text-ink flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-blue-500" />
                    Answer
                  </h2>
                  <span className="font-mono text-xs text-muted bg-white/5 px-2 py-1 rounded-md">
                    {response.latency_ms}ms
                  </span>
                </div>
                <div className="prose prose-invert prose-p:leading-relaxed prose-p:text-body max-w-none">
                  <p className="whitespace-pre-wrap">{response.answer}</p>
                </div>
              </div>

              {/* Sources Section */}
              {response.sources && response.sources.length > 0 && (
                <div className="space-y-6">
                  <h3 className="font-display font-semibold text-lg text-ink px-2">
                    Sources ({response.sources.length})
                  </h3>
                  <div className="grid gap-4">
                    {response.sources.map((source, idx) => (
                      <motion.div
                        key={idx}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: idx * 0.1 }}
                        className="glass-card p-6 flex flex-col gap-3 group"
                      >
                        <div className="flex items-start justify-between gap-4">
                          <div className="flex-1">
                            <h4 className="font-display font-medium text-sm text-ink mb-1 group-hover:text-blue-400 transition-colors">
                              {source.section_heading}
                            </h4>
                            <div className="flex items-center gap-2">
                              <div className="w-16 h-1 bg-white/10 rounded-full overflow-hidden">
                                <div className="h-full bg-blue-500" style={{ width: `${source.relevance_score * 100}%` }} />
                              </div>
                              <p className="font-mono text-[10px] text-muted">
                                {(source.relevance_score * 100).toFixed(1)}% match
                              </p>
                            </div>
                          </div>
                          <a
                            href={source.source_url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="shrink-0 w-8 h-8 flex items-center justify-center rounded-full bg-white/5 text-muted hover:bg-white/10 hover:text-ink transition-all"
                          >
                            <ExternalLink className="w-4 h-4" />
                          </a>
                        </div>
                        <p className="font-text text-sm text-muted line-clamp-2 mt-2">
                          {source.text}
                        </p>
                      </motion.div>
                    ))}
                  </div>
                </div>
              )}

              {/* Metadata */}
              <div className="flex items-center gap-4 px-2">
                <div className="font-mono text-[10px] uppercase tracking-wider text-muted px-3 py-1.5 rounded-full border border-hairline bg-surface-soft">
                  Engine: {response.backend_used}
                </div>
                {!response.context_found && (
                  <div className="font-mono text-[10px] uppercase tracking-wider text-amber-500 px-3 py-1.5 rounded-full border border-amber-500/20 bg-amber-500/10">
                    ⚠ Low confidence
                  </div>
                )}
              </div>
            </motion.div>
          )}
        </div>
      </section>
    </div>
  );
}
