import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Link from "next/link";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Colophon — Claude API Knowledge Assistant",
  description:
    "Grounded answers to Claude API questions — every claim cited to real documentation.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${geistSans.variable} ${geistMono.variable}`}>
      <body className="bg-canvas text-ink min-h-screen flex flex-col">
        {/* Glassmorphic Navigation */}
        <nav className="fixed top-0 w-full z-50 flex items-center justify-between px-8 py-4 backdrop-blur-md bg-canvas/60 border-b border-hairline">
          {/* MENU Label (Left) */}
          <div className="font-mono text-xs uppercase tracking-widest text-muted">
            MENU
          </div>

          {/* Wordmark - Center */}
          <Link href="/" className="text-xl font-display font-bold tracking-tight text-ink hover:text-white/80 transition-colors">
            Colophon
          </Link>

          {/* Navigation Links (Right) */}
          <div className="flex items-center gap-6">
            <Link href="/ask" className="font-medium text-sm text-muted hover:text-ink transition-colors">
              Ask
            </Link>
            <Link href="/evaluation" className="font-medium text-sm text-muted hover:text-ink transition-colors">
              Evaluation
            </Link>
            <Link href="/architecture" className="font-medium text-sm text-muted hover:text-ink transition-colors">
              Architecture
            </Link>
          </div>
        </nav>

        {/* Main Content */}
        <main className="flex-1 w-full">{children}</main>

        {/* Modern Footer */}
        <footer className="w-full bg-canvas/80 border-t border-hairline py-12 px-8 mt-auto backdrop-blur-md">
          <div className="max-w-6xl mx-auto flex flex-col md:flex-row items-center justify-between gap-6">
            <p className="font-text text-sm text-muted leading-relaxed max-w-md text-center md:text-left">
              Colophon is an independent portfolio project grounded in Anthropic Claude documentation. 
              Every answer is cited to real sources.
            </p>
            <div className="flex items-center justify-between gap-4">
              <div className="font-medium text-xs text-muted-soft">
                © 2026 · All rights reserved
              </div>
              <div className="font-display font-bold text-sm text-ink/50 bg-white/5 rounded-full w-8 h-8 flex items-center justify-center">
                C
              </div>
            </div>
          </div>
        </footer>
      </body>
    </html>
  );
}
