---
description: Explicitly invoked expensive deep architecture and production-risk review using GPT-6 Astra, without modifying files.
agent: deep-review
model: openai/gpt-6-astra
---

The user has explicitly invoked `/deep-review`. Perform a deep read-only senior production engineering review of the actual repository. Follow AGENTS.md. Additional user context: $ARGUMENTS

Use read/search tools and `readonly_git` for inspection. Direct shell, verification, editing, and delegation tools are denied by this agent's permissions.

This expensive/high-intelligence review is opt-in only. Never recommend automatically invoking it, route routine commands to this model, or use this command as an automatic workflow step. Do not modify files, install dependencies, commit, deploy, run mutating shell commands, or make paid/live service calls. Keep secrets and private data out of output.

Map system architecture and end-to-end data flows from implementation, tests, configuration, dependencies, and deployment files. Analyze:
- Architecture boundaries, coupling, and major missed product/engineering opportunities.
- Hidden correctness issues, edge cases, invariants, and data integrity.
- Concurrency, shared state, cache consistency, and multi-process behavior.
- Reliability under partial failure, retries, idempotency, timeouts, cancellation, recovery, and dependency outages.
- Security/privacy, authorization boundaries, secret handling, and external integrations.
- Performance/scalability, resource limits, backpressure, latency, and API/token/cost behavior.
- Production deployment risks, configuration, persistence, migrations, rollback, observability, and operational diagnosis.
- Existing tests, regression gaps, and evidence needed to validate critical assumptions.

Verify every finding against actual repository code; cite exact repository-relative files/lines and trace the trigger or failure scenario. Check existing protections before declaring a problem. Do not invent vulnerabilities, provider guarantees, benchmark results, test outcomes, or speculative rewrite requirements. Separate confirmed bugs, optional improvements, and unresolved assumptions.

Return an architecture summary, then findings ordered P0 correctness/security, P1 production reliability, P2 performance/scalability, P3 maintainability/polish. Each finding must include evidence, current problem, production impact, and a practical fix. Include verification limits, deployment blockers, and a prioritized remediation sequence. Finish with 1-3 genuinely valuable, code-supported product/engineering improvements. Do not pad the report with cosmetic changes.
