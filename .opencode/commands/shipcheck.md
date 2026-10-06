---
description: Execute relevant repository verification and obtain a final production review before issuing a shipping verdict.
agent: shipcheck
model: openai/gpt-6.1-sol
---

Perform an evidence-based production shipping check. Follow AGENTS.md. Additional user context: $ARGUMENTS

Use `readonly_git` for git inspection and `readonly_verify` for verification commands. Direct shell and file-edit tools are denied. Verification runs in an OS-enforced sandbox: reads are limited to the repository, its disposable temporary directory, and explicit system/runtime paths; writes are allowed only in that temporary directory; network access is denied. It cannot read unrelated user files or credentials outside the repository. If the sandbox is unavailable or a check requires forbidden reads/writes/network, report it as BLOCKED; never fall back to an unrestricted shell. It uses a minimal environment with HOME/config/cache/temp locations redirected outside the repository.

Do not modify application code or configuration, apply autofixes, install/update dependencies, commit, push, deploy, run destructive operations, or invoke the expensive deep-review model. Verification may create normal disposable test/build artifacts, but must not overwrite tracked files or use real production data or services. Do not expose secrets.

1. Inspect git status, both staged and unstaged diffs (disable external diff/text conversion), and relevant untracked files. Record the change scope and initial worktree state.
2. Determine relevant test/lint/typecheck/build commands from the actual repository: AGENTS.md, Makefile, .github/workflows/ci.yml, requirements*.txt, tests, Dockerfile, docker-compose.yml, and any other applicable configuration. Read commands before executing them to understand side effects. Current starting points are `make test` and `make lint`; do not assume a typecheck target or build tool exists.
3. Actually execute the appropriate checks using the available environment, with sensible timeouts. Include meaningful regression checks for the affected paths. Use isolated test data and mocks where supported. Do not launch app.py, agent.py, deployment scripts, ingestion jobs, or paid/live evaluations as substitutes for tests. If required tooling, credentials, infrastructure, or permissions are unavailable, record the affected check as blocked rather than passing. Do not weaken or repair checks just to obtain a passing result.
4. Capture each exact command, exit status (when available), and concise result. Distinguish PASS, FAIL, BLOCKED, and NOT APPLICABLE. Report coverage/behavior limits and failures attributable to the environment separately from verified code defects. Never say a check passed unless it actually ran and succeeded.
5. Use the Task/delegation tool with the `production-reviewer` subagent for a final review. Provide the complete change scope, relevant files, exact verification results, and unresolved risks. Ask it to inspect actual code/diffs, prioritize bugs/regressions/security/concurrency/performance/reliability/missing tests, and return P0-P3 findings with file/line evidence and fixes. The reviewer must not edit files. If the subagent cannot be invoked, record the final review as blocked.
6. Re-inspect git status/diff to detect unexpected tracked-file changes. Do not discard existing user work or automatically clean artifacts. Identify blocking defects and required verification gaps before production.

Finish with exactly one verdict heading:
- `READY TO SHIP`: all required applicable checks actually passed, the final reviewer completed, and no blocking findings or unresolved required checks remain.
- `NOT READY TO SHIP`: any required check failed or was blocked/not run, final review was unavailable, or blocking risks remain.

Below the heading include reasons, the verification-results table, reviewer findings with priorities/evidence, remaining blockers and recommended actions, and worktree changes observed. A passing test suite alone does not prove production readiness.
