---
description: Identify the five highest-value verified production improvements across the application without editing files.
agent: plan
model: openai/gpt-6.1-sol
---

Analyze this whole application as a senior production engineer. Follow AGENTS.md. Do not modify files, run mutating commands, commit, or deploy. Additional user context: $ARGUMENTS

Understand the actual architecture and data flow by reading relevant implementation, tests, configuration, dependency files, and deployment files. Inspect all major request, ingestion, persistence, retrieval, provider, and failure paths. Do not rely on README claims or previous review conclusions without verifying them against current code. Keep secrets and private user data out of the report.

Return the five highest-value production improvements, ordered by priority and impact. Prioritize correctness, security/privacy, reliability, architecture, concurrency/state consistency, retries/idempotency, data integrity, performance/scalability, caching, observability, latency/API/token cost, deployment, and developer experience. Do not suggest cosmetic refactors or speculative rewrites.

For each finding:
- Verify it against actual code and cite repository-relative files and line numbers.
- Distinguish a bug from an optional improvement; identify the trigger or failing scenario.
- Explain the current problem and production impact.
- Propose a practical fix consistent with existing abstractions.
- Classify P0 correctness/security, P1 production reliability, P2 performance/scalability, or P3 maintainability/polish.

Do not invent problems to reach five; if fewer than five are supported, explicitly state the evidence limit. State what was inspected, what was actually executed, and what remains uncertain. End with a recommended implementation order. This command must not invoke `/deep-review` or select its expensive model.
