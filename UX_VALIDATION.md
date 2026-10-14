# Demo visitor-journey validation

## Automated result

The local synthetic end-to-end journey passed on **1360px desktop** and **390px
mobile-emulated** Chrome layouts, with a separate browser profile for identity
isolation. This verifies application interactions, not live inference quality or
hosted-service configuration.

The actual Gradio UI, LangGraph agent, source registry, page renderer, same-origin
private-session routes and encrypted cookies execute. Provider responses,
retrieval data, CAPTCHA challenge completion, memory extraction and the remote
memory service are synthetic. No workspace credentials, real private memory,
private documents, or hosted database rows are used.

## Reproducible commands

With repository dependencies, Selenium, and Chrome installed:

```bash
# Entire journey, including memory and recovery states.
PYTHONDONTWRITEBYTECODE=1 python -m tests.browser_demo_ux \
  --full-journey --artifacts "$TMPDIR/opencode/phase4-browser"

# Private memory and two simultaneous identities without repeating public checks.
PYTHONDONTWRITEBYTECODE=1 python -m tests.browser_demo_ux \
  --memory-only --artifacts "$TMPDIR/opencode/phase4-browser"

# Existing public research/interaction checks.
PYTHONDONTWRITEBYTECODE=1 python -m tests.browser_demo_ux \
  --artifacts "$TMPDIR/opencode/phase4-browser"

PYTHONDONTWRITEBYTECODE=1 make test
make lint
make test-db
git diff --check
docker build -t agenticrag-ux-phase4-check:local .
docker run --rm --network none --entrypoint python \
  agenticrag-ux-phase4-check:local -m pip check
```

Use an existing temporary parent for artifacts. `--chrome` selects another Chrome
binary when needed. Selenium is an optional verification dependency, not a runtime
dependency. Screenshots and disposable browser profiles stay outside the repository.

## Coverage matrix

| Visitor action or condition | Verification | Result |
|---|---|---|
| Arrival, scope, starter tasks, no required account | Browser on both layouts | Passed |
| Report titles/dates/coverage and original-PDF links | Browser source cards | Passed |
| Fact, comparison and boundary examples use correct options | Real graph with synthetic provider | Passed |
| Live processing stage and optimized queries | Browser execution details | Passed |
| Numbered citations and rejection of unknown IDs | Browser + regression suite | Passed |
| Selected source page loads through opaque route | Actual synthetic PNG HTTP response | Passed |
| Earlier source remains selectable after follow-up | Browser links and card buttons | Passed |
| Follow-up preserves context/settings without generating suggestions | Browser + regression suite | Passed |
| Empty corpus and unavailable corpus pause controls | Browser recheck/recovery | Passed |
| Availability checks do not request generation | Synthetic provider call tracking | Passed |
| Missing evidence yields an explanation with no invented reference | Browser | Passed |
| Provider overload and successful retry | Browser | Passed |
| Disabled/unavailable checks are distinct from successful checking | Browser + strict-schema tests | Passed |
| Private-session start updates UI after cookie creation | Actual JS/HTTP/Gradio callback | Passed |
| Cookie is encrypted, HttpOnly, SameSite=Lax | Browser cookie assertions | Passed |
| Save prerequisites, immediate panel update, and reload | Actual UI with synthetic memory adapter | Passed |
| Forget clears cookie, memory panel, evidence and execution details | Browser | Passed |
| Two active identities retain distinct saved memories | Independent browser profiles | Passed |
| Forget in one profile preserves the other profile's memory | Browser reload + adapter state | Passed |
| Return to public mode clears context without claiming remote deletion | Browser + adapter state | Passed |
| Late private answers/previews/progress cannot restore ended context | Concurrent regression tests | Passed |
| RLS ownership, revisions, close markers and anti-resurrection | 28 pgTAP checks on disposable PostgreSQL 17.11 | Passed |
| Keyboard controls, layout, and question-placeholder contrast | Browser; mobile activation by keyboard | Passed |

## Results actually recorded

- Full browser journey: passed on desktop, mobile-emulated layout, and an
  independent browser profile. A fresh renderer/profile is used per viewport.
- `PYTHONDONTWRITEBYTECODE=1 make test`: **116 passed**, 40 dependency/fixture warnings.
- `make lint`: **0 configured flake8 issues**.
- `make test-db`: **28 pgTAP assertions passed** on disposable PostgreSQL 17.11.
  This uses a minimal Auth UID/role fixture, not hosted GoTrue/PostgREST.
- Production image build and network-disabled `pip check`: passed, with the
  existing Docker keyword-casing warning.
- Two confirmed UI regressions were corrected: persistent loading overlays from
  multi-component identity refreshes, and JS-only session actions not reliably
  invoking their `.success()` refresh in Gradio 6.9. Actions now perform the async
  same-origin request and backend identity refresh within one event. Backend
  authorization remains unchanged.

## Remaining live checks before a hosted release

- [ ] Launch the intended hosted revision with the approved IPCC corpus enabled.
- [ ] Run the supported research presets against the real generation provider;
      inspect citation correctness, dates, uncertainty and evidence coverage.
- [ ] Complete actual Turnstile verification and Supabase anonymous Auth on the
      direct HTTPS host; verify Secure/HttpOnly cookies and configured Origin.
- [ ] Save, reload, Forget and return-to-public flows with two real browser
      identities on that host, using synthetic conversations only.
- [ ] Verify the intended mobile/touch devices and additional browser engines.
- [ ] Record observed provider/quota failures and hosted recovery behavior.
- [ ] Run a science-specific quality benchmark before publishing quality claims
      for this corpus. Retrieval smoke checks are not accuracy guarantees.

These checks are **pending**, not passing. The local checklist completion is not
a hosted production-readiness claim. No push, cloud deployment, live CAPTCHA,
live Supabase mutation or live inference was performed during this validation.
