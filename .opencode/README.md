# Read-only production workflows

- `/shipcheck`: dedicated `shipcheck` agent, `openai/gpt-6.1-sol`; runs sandboxed verification and delegates its final review only to `production-reviewer`.
- `/deep-review`: dedicated `deep-review` agent, `openai/gpt-6-astra`; explicit opt-in analysis, without verification or delegation.
- `production-reviewer`: retains its model and review instructions; shell inspection now uses the same sandboxed read-only git tool.

Each agent defaults to denying every tool, then allows only its required read/search and restricted tools. Direct shell execution, editing/patching, other custom/MCP tools, and delegation to unrestricted agents are denied.

`tools/readonly.ts` provides `readonly_git` (exact status/diff/log commands) and `readonly_verify` (verification for shipcheck only). Both use macOS `sandbox-exec`; the sandbox denies filesystem writes, network access, and helper-service access. Verification alone permits writes to a fresh external temporary directory, redirects common cache/temp locations there, and removes it afterward. Restrictions apply to child processes and filesystem targets reached through symlinks. Repository-local build outputs and live-service checks are blocked; report these as required verification gaps rather than bypassing the sandbox.

Filesystem reads are denied by default and allowed only within the canonical repository, that invocation's temporary directory, and the explicit system/runtime allowlist in `tools/readonly.ts`. It covers macOS executables/libraries, developer tools, and standard Homebrew/Anaconda installation subtrees, plus individual runtime files/devices. It does **not** grant entire `/System` (whose Data volume exposes user files), `/Library`, `/opt`, `/private/var`, or user home directories. Ancestor paths receive metadata-only access for pathname traversal; a literal `/` directory grant is required by macOS's dynamic loader and does not allow reading its descendants.

Unrelated user files (SSH/AWS/config credentials, browser data, other projects, and other temporary directories) remain unreadable even through symlinks or child processes. Verification gets a minimal environment, a fixed system/repository-venv PATH, and temporary HOME/XDG directories; inherited credentials, user startup files, loader overrides, and user package paths are not passed through. External user-installed runtimes, credentials, or fixtures that require broader access are blocked rather than implicitly allowed. Repository-local secrets are within the requested repository read boundary; this tool does not redact them.

The tools fail closed if the platform or sandbox is unavailable. macOS is currently required for execution; other platforms still support read/search analysis. No unsandboxed fallback is provided.

Validate with Node.js 24+ and the installed OpenCode CLI:

```sh
node --test .opencode/tests/readonly.test.mjs
opencode debug agent shipcheck
opencode debug agent deep-review
opencode debug agent production-reviewer
```

Tests check resolved permissions, command/model bindings, delegation restrictions, external sentinel-secret read denial (including symlinks, hardlinks, child processes, and Data-volume aliases), normal repository reads/verification, sandbox write denial, permitted disposable artifacts, environment isolation, network denial, timeouts, and fail-closed behavior. Synthetic secrets are created outside the repository; actual user credentials are not inspected. Tests do not invoke either model or run an application review.

Quit and restart OpenCode after changing this configuration; existing sessions keep their previously loaded configuration.
