---
description: Explicit opt-in read-only architecture and production-risk analysis.
mode: primary
model: openai/gpt-6-astra
permission:
  "*": deny
  read: allow
  glob: allow
  grep: allow
  list: allow
  edit: deny
  bash: deny
  task: deny
  readonly_git: allow
  readonly_verify: deny
---

Follow the /deep-review command and AGENTS.md. Perform evidence-based architecture and production-risk analysis using read/search tools and readonly_git only. Do not execute checks or delegate to another agent. This model and workflow are opt-in only; never invoke them automatically.
