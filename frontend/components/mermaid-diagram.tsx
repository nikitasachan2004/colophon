"use client";

import * as React from "react";
import mermaid from "mermaid";

interface MermaidDiagramProps {
  chart: string;
  className?: string;
}

let initialized = false;

export function MermaidDiagram({ chart, className }: MermaidDiagramProps) {
  const ref = React.useRef<HTMLDivElement>(null);
  const [svg, setSvg] = React.useState<string>("");
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    if (!initialized) {
      mermaid.initialize({
        startOnLoad: false,
        theme: "neutral",
        fontFamily: "inherit",
        fontSize: 13,
      });
      initialized = true;
    }

    const id = `mermaid-${Math.random().toString(36).slice(2)}`;
    mermaid
      .render(id, chart)
      .then(({ svg }) => setSvg(svg))
      .catch((err) => setError(String(err)));
  }, [chart]);

  if (error) {
    return (
      <pre className="rounded-lg border border-destructive/40 bg-destructive/5 p-4 text-xs text-destructive overflow-auto">
        Diagram error: {error}
      </pre>
    );
  }

  if (!svg) {
    return (
      <div className="flex h-64 items-center justify-center rounded-lg border border-border bg-muted/30">
        <p className="text-sm text-muted-foreground">Rendering diagram…</p>
      </div>
    );
  }

  return (
    <div
      ref={ref}
      className={className}
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  );
}
