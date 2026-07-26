# Rules Document
## Production RAG Knowledge Assistant — Working Rules & Guardrails

These are binding constraints for how this project gets built — for you, and for any AI coding assistant (Claude Code or otherwise) helping implement it. If a suggestion conflicts with a rule here, the rule wins.

---

## 1. Scope Discipline

- **No feature is added that isn't in `projectrequirement.md` §4 (In Scope) without first updating that document.** This project has a hard time budget (60–80 hrs); scope creep is the single biggest risk to actually finishing.
- Anything in §5 (Out of Scope) — auth, multi-tenancy, file uploads, fine-tuning, multi-language, streaming UI — stays out, even if it seems like "just a small addition." Log the idea in `memory.md` under "Deferred Ideas" instead of building it.
- If a phase in `phases.md` is running over budget, cut the *stretch* items in that phase first (marked "Could" priority in the requirements), never cut the evaluation phase (Phase 4).

## 2. Legal / Ethical Constraints (non-negotiable)

- Only scrape publicly accessible documentation pages. Respect `robots.txt`. No authentication bypass, no scraping behind a login.
- Do not scrape or store any personal data, customer data, or anything that isn't the company's own published API documentation.
- The deployed app must clearly disclose it is an independent, unaffiliated portfolio project (see `design.md` §8 footer requirement) — never imply official endorsement by the target company.
- Cache scraped content locally rather than re-fetching on every ingestion run, to be a good citizen of the target site's servers.

## 3. Security

- No API keys, tokens, or secrets are ever committed to git. `.env` is gitignored from commit 1. `.env.example` (with placeholder values only) is the only env file that's committed.
- Add a pre-commit hook or CI check that scans for accidentally committed secrets before every push.
- The `/admin/reingest` endpoint (design.md §9) is never exposed in the public deployment — dev-only, feature-flagged off by default in the prod config.

## 4. Code Quality Standards

- Every function in `src/` that contains non-trivial logic (chunking boundaries, RRF fusion, confidence threshold logic) gets a unit test — no exceptions, because these are exactly the parts an interviewer will ask you to explain and where subtle bugs hide.
- Type hints on all function signatures in `src/` (this is a "production" project — type safety is part of that claim, not decoration).
- No hardcoded magic numbers for tunable parameters (chunk size, top-k, thresholds) — they live in `config.py` / `.env`, per `design.md` §7, so they can be tuned during Phase 4 without code changes.
- Every module has a short docstring stating what it does and why (not just what — the *why* is what makes the README/code review land well).

## 5. Honesty Rules (this is the rule that most affects your resume claim)

- **Every metric that ends up in the README or on your resume must come from an actual logged evaluation run, not an estimate, a "should be about," or a single lucky test.** `phases.md` Phase 4 exists specifically to produce these numbers for real.
- If a component underperforms (e.g., reranker too slow on free hosting, faithfulness score lower than hoped), document it honestly as a known limitation rather than hiding it. A well-documented trade-off is a *stronger* interview answer than a suspiciously perfect project with no acknowledged weaknesses.
- The dev/prod backend split (Ollama/Groq) must be disclosed clearly in the README — don't let a reader assume the deployed version is running a local model if it's actually calling Groq.

## 6. Working With an AI Coding Assistant on This Project

- Before implementing any file, the assistant should re-read the relevant section of `design.md` — this rules file exists precisely so an assistant doesn't "helpfully" reintroduce out-of-scope features or skip the evaluation phase to move faster.
- After any significant decision (a library swap, a parameter chosen after tuning, a scope cut), **log it in `memory.md`** immediately — don't rely on conversation history alone, since sessions will span the full 6 weeks and context will be lost between them.
- If asked to skip tests "to move faster," refuse and flag it — per NFR-5 in `projectrequirement.md`, test coverage on core logic is a hard requirement, not optional polish.
- If a proposed change would violate §1 (scope) or §5 (honesty), the assistant should say so explicitly rather than silently complying.

## 7. Git / Commit Discipline

- Commit early and often with meaningful messages — your commit history is itself evidence of iterative engineering, and a recruiter or interviewer skimming the repo may look at it.
- Each phase in `phases.md` should correspond to a rough cluster of commits, ideally tagged or noted, so the repo tells the story of the build, not just the final state.
- No single giant "final commit" that dumps the whole project at once — this is a common student-project red flag noted in `projectrequirement.md`'s underlying research (half-finished-looking repos hurt more than they help).
