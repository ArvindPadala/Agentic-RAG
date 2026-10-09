# Engineering Instructions

## Before making changes
- Understand the existing architecture and data flow first.
- Read the relevant implementation, tests, configuration, and deployment files.
- Prefer existing abstractions over creating duplicate systems.
- Do not make speculative changes.

## Implementation quality
For every meaningful change, consider:
- correctness and edge cases
- failure handling and retries
- concurrency and state consistency
- security, privacy, and secrets
- performance and scalability
- latency, API, token, and cost impact
- caching and cache invalidation
- observability and logging
- backward compatibility
- deployment impact

## Verification
Before saying a task is complete:
- inspect the final git diff, including staged changes and relevant untracked files
- run relevant tests
- run lint/type checks/build checks when available
- verify the requested behavior
- check likely regressions

Never claim something was tested if it was not actually tested.

## Production review
When you discover problems or improvement opportunities:
- verify them against the code
- distinguish bugs from optional improvements
- rank them:
  - P0: correctness/security
  - P1: production reliability
  - P2: performance/scalability
  - P3: maintainability/polish
- include file/line evidence, production impact, and a recommended fix
- separate verified findings from assumptions or checks that could not be run

## Proactive improvements
After substantial work, mention 1-3 high-value improvements you noticed that would materially improve the application.

Do not recommend unnecessary rewrites or refactors just for style.

## OpenCode production workflow
- `/improve`: read-only whole-application analysis of the five highest-value production improvements.
- `/review`: read-only review of the current git diff by the `production-reviewer` subagent.
- `/shipcheck`: run relevant verification and obtain a final production review; report `READY TO SHIP` or `NOT READY TO SHIP` with evidence.
- `/deep-review`: expensive architecture and production-risk analysis using `openai/gpt-6-astra`. Run only when the user explicitly invokes it; never select this model or command automatically.
- Review commands must not edit files. `/shipcheck` must not modify application code, apply autofixes, update dependencies, commit, or deploy.
- Discover verification commands from the repository rather than assuming them. Current starting points are `make test` and `make lint` in `Makefile`, `.github/workflows/ci.yml`, `requirements*.txt`, `Dockerfile`, and `docker-compose.yml`.
- Report exact commands and results; identify skipped, unavailable, or blocked checks explicitly. Missing required verification is a shipping blocker, not a passing check.
- Keep secrets, private document contents, and personal conversation memory out of review output.
