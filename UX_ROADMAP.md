# Demo UX implementation checklist

## Goal

Help a first-time visitor **understand the reports → ask a useful question →
inspect the answer's evidence → ask a follow-up** without learning RAG settings
or creating an account first.

Implement the numbered items one by one, in order. Check off an item only after
its behavior is implemented and verified. Record exact checks and any remaining
limits in the progress log below.

Priority labels: **P1** = production reliability; **P3** = usability/polish.
Implementation order also considers dependencies and visitor impact.

## Phase 1 — First-time visitor experience

- [x] **01. Make the landing message outcome-focused.** *(P3)*
  Explain what visitors can research, that answers use three dated IPCC reports,
  and that public chat needs no sign-up. Use a compact header and a simple
  “Explore reports / How this works” navigation hierarchy.
  **Done when:** visitors can identify the purpose, scope, and first action from
  the initial view; technical branding does not dominate the introduction.

- [x] **02. Add corpus-aware readiness and unavailable states.** *(P1)*
  Derive availability from the actual authorized corpus. Distinguish an
  unavailable/empty corpus from a question with insufficient evidence. Do not
  advertise a usable showcase when its reports are unavailable.
  **Done when:** the page explains unavailable states clearly and does not send
  ordinary research requests to the LLM when no authorized documents are ready.

- [x] **03. Add a browsable report collection.** *(P3)*
  Show one card per approved report: title, publisher, year, topic coverage, and
  original-PDF link. Keep report dates and scope visible.
  **Done when:** visitors can explore all three reports and understand what
  questions they support without asking the model or uploading files.

- [x] **04. Replace the blank chat with a useful empty state.** *(P3)*
  Put a short welcome and starter questions in the main research area. Give chat
  the primary visual emphasis; avoid a large empty evidence panel on arrival.
  **Done when:** the initial workspace has a clear action and the welcome state
  transitions cleanly into a conversation and returns after clearing chat.

- [x] **05. Make examples task-oriented and configure them automatically.** *(P3)*
  Lead with “Find a fact,” then “Compare evidence,” then “Explore a pathway.”
  Explain each example's purpose in ordinary language. Presets must set their
  actual search/decomposition/check options without requiring manual toggles.
  **Done when:** each example runs with the advertised settings and a visitor
  can start a useful research task with one explicit action.

- [x] **06. Move the negative example into “Test the system's limits.”** *(P3)*
  Keep the unsupported city/date forecast as a secondary boundary demo after
  the positive examples. Explain that acknowledging missing evidence is useful.
  **Done when:** the first suggested experience is a supported question, while
  the boundary example remains easy to find and run.

- [x] **07. Collapse technical controls behind progressive disclosure.** *(P3)*
  Keep a sensible default retrieval configuration. Move vector/RRF/reranker,
  decomposition, and guardrail controls into labeled advanced options, with
  plain-language descriptions and accurate latency/call tradeoffs.
  **Done when:** ordinary visitors can use the demo without choosing an engine;
  technical visitors can still inspect and change every existing option.

- [x] **08. Introduce memory at the point of usefulness.** *(P3)*
  Move private-session controls out of the main onboarding path. Present an
  optional “Remember my research interests” action, explain temporary-session
  limits, and make Save Memory prerequisites and session status clear.
  **Done when:** public chat remains frictionless, memory actions have a clear
  lifecycle, and the existing ownership/Forget/stale-response protections hold.

- [x] **09. Make the research workspace responsive and accessible.** *(P3)*
  Give the conversation most space on desktop and reveal evidence on demand.
  Stack or expand evidence beneath answers on small screens. Check keyboard
  navigation, control labels, focus behavior, text contrast, and readable sizing.
  **Done when:** desktop/mobile views support asking and inspecting evidence
  without clipped controls or reliance on hover-only interactions.

## Phase 2 — Evidence-first answers

- [x] **10. Preserve structured retrieval/citation data outside model prose.** *(P1)*
  Retain authorized source IDs, report titles, excerpts, page positions, bounds,
  and image/PDF links as request-scoped data. Distinguish retrieved passages from
  citations actually referenced in an answer; validate citation references.
  **Done when:** evidence remains available even if the model omits an image URL,
  fabricated source IDs cannot become valid citations, and session isolation is
  preserved. This is the foundation for items 11–14.

- [x] **11. Add compact, numbered citations to answers.** *(P3)*
  Map references such as [1] and [2] to the structured evidence records. Retain
  report dates and clarify PDF file-page positions versus printed page labels.
  **Done when:** each valid citation identifies its supporting passage and can
  open the corresponding evidence without exposing raw internal paths.

- [x] **12. Replace unlabeled images with evidence cards.** *(P3)*
  Show report title, year, PDF page, supporting excerpt, and original-PDF link.
  Use explicit “Retrieved evidence” versus “Cited in this answer” labels where
  appropriate. Do not treat an arbitrary retrieval score as factual confidence.
  **Done when:** visitors understand which report/page a passage comes from and
  why it is shown before expanding its image.

- [x] **13. Open the selected citation's highlighted PDF page on demand.** *(P3)*
  Link each evidence card/citation to its specific rendered page. Support a
  readable expanded preview alongside chat on desktop and below it on mobile.
  **Done when:** selecting a reference displays the correct report and page,
  and visitors can inspect the highlight and return to the answer easily.

- [x] **14. Keep evidence associated with its original answer.** *(P1)*
  Store evidence per completed turn instead of replacing all context with the
  latest answer's images. Clear or invalidate it with the conversation/session
  lifecycle; keep the state bounded.
  **Done when:** an earlier answer's citations still open its own evidence after
  follow-ups, and old private-session outputs cannot reappear in a new session.

- [x] **15. Make follow-up research easy.** *(P3)*
  Offer a clear follow-up input and useful continuation prompts tied to the
  supported research tasks. Prefer curated suggestions initially rather than
  adding an LLM request solely to generate UI suggestions.
  **Done when:** a visitor can continue from an answer without resetting context
  or accidentally launching a different preset/settings combination.

## Phase 3 — Honest engineering visibility

- [x] **16. Show actual request progress and recovery actions.** *(P1)*
  Expose real queue/search/comparison/generation/check events, where applicable.
  Distinguish waiting, provider failure, and insufficient evidence; provide clear
  retry guidance. Do not simulate stages or completion percentages.
  **Done when:** a waiting visitor can tell what is happening, and failures do
  not look like successful research or silently damage conversation state.

- [x] **17. Make query decomposition observable.** *(P3)*
  Display actual generated search sub-queries in expandable execution details,
  including fallback/not-used status. Show operational facts, not hidden model
  reasoning. Keep private-session details scoped to that session.
  **Done when:** the decomposition demo visibly demonstrates its real behavior
  and the UI no longer promises steps it cannot show.

- [x] **18. Give the faithfulness check an explicit outcome contract.** *(P1)*
  Represent Not run, Completed, Flagged, and Unavailable distinctly. Validate
  provider output and stop interpreting API/parse failures as a successful check.
  Label this as an automated grounding check, not a guarantee of correctness.
  **Done when:** UI status matches the actual check result, including errors and
  disabled checks, and the answer remains transparently inspectable.

- [x] **19. Add expandable execution details for technical visitors.** *(P3)*
  Show actual retrieval mode, search count, retrieved/cited source counts, timing,
  decomposition status, and grounding-check status. Keep architecture/evaluation
  content secondary to the main research workspace.
  **Done when:** visitors can inspect the system after experiencing its value,
  without raw credentials, internal paths, other users' state, or invented stats.

- [x] **20. Keep performance and quality claims tied to their dataset.** *(P1)*
  Preserve the distinction between historical metrics, IPCC retrieval smoke
  checks, and future science-corpus quality measurements. Make that distinction
  visible wherever metrics are presented in the redesigned demo.
  **Done when:** visitors cannot mistake earlier-corpus scores or a successful
  grounding check for a measured guarantee about the current science corpus.

## Phase 4 — End-to-end visitor validation

- [x] **21. Validate the complete first-time visitor journey.** *(P1/P3)*
  Exercise arrival → report exploration → supported example → cited answer →
  evidence inspection → follow-up → optional memory → clear/end session.
  Cover empty-corpus, missing-evidence, provider/check failure, older citations,
  and session switching. Verify desktop/mobile and keyboard interactions.
  **Done when:** relevant regression checks pass and actual browser smoke results
  are recorded. Missing live/browser checks must be stated explicitly.

## Implementation constraints

- Reuse the existing Gradio UI, agent, authorization boundaries, and session
  lifecycle rather than introducing a separate frontend/backend system.
- Keep public exploration account-free and hosted public uploads disabled until
  their authorization/ingestion milestone is implemented.
- Initial report descriptions and examples should be curated static content,
  adding no LLM calls. Extra processing must be deliberate and visible.
- Keep evidence and execution details scoped to the requesting session; do not
  log personal conversations or expose credentials/private data.
- Preserve original report links, dates, uncertainty qualifiers, and extraction
  limits. Existing authorization applies even when controls are hidden.
- Verify each meaningful increment with appropriate tests/checks and inspect its
  final diff before marking it complete.

## Product and UX references

- [Google curated/featured notebooks](https://support.google.com/notebooklm/answer/16322204?hl=en):
  a ready-to-use source collection for exploration.
- [Google source-grounded chat](https://support.google.com/notebooklm/answer/16179559?hl=en):
  inspect cited quotes in their original context.
- [ChatPDF](https://www.chatpdf.com/): side-by-side chat/document inspection and
  clickable PDF citations.
- [Perplexity](https://www.perplexity.ai/help-center/en/articles/10352895-how-does-perplexity-work):
  compact numbered citations and conversational follow-ups.
- [Elicit](https://elicit.com/solutions/literature-review): task-oriented research
  workflows and source-supported evidence synthesis.
- [NN/G progressive disclosure](https://www.nngroup.com/articles/progressive-disclosure/):
  core actions first, specialized controls on request.

These are interaction references; their documentation does not establish that
their underlying retrieval architecture matches this application's.

## Progress log

- **Phase 1 implemented (items 01–09).** The landing prioritizes a research task,
  shows authorized corpus readiness, and links to report cards. The welcome
  includes three supported presets that set their actual retrieval/options state.
  The empty chat is hidden until a response; clearing or switching sessions
  restores the welcome. Source pages, advanced options, boundary testing, and
  research memory use keyboard-operable accordions. Save is enabled only after
  conversation context and a private identity exist (or for the local workflow).
- Availability is enforced again in the chat handler before tool construction,
  decomposition, generation, or checking; private/revoked chunks never count as
  public readiness. Rechecking availability updates disabled controls and cards
  without an LLM call. Backend memory authorization/lifecycle guards are retained.
- Verification: `PYTHONDONTWRITEBYTECODE=1 make test` — 78 passed, 28 dependency/
  fixture warnings; `make lint` — 0 configured issues. The suite exercises real
  filtered Chroma readiness/revocation, skipped provider calls, recovery controls,
  actual Gradio preset/state/clear callbacks, memory save eligibility, ownership,
  and suppression of late private outputs.
- Real Chrome verification:
  `PYTHONDONTWRITEBYTECODE=1 python -m tests.browser_demo_ux --artifacts <temporary-directory>`
  — desktop (1360px) and mobile-emulated (390px) flows passed: presets/settings,
  welcome/clear, loaded synthetic PDF-page images, keyboard accordions, original
  report links, question-placeholder contrast (at least 4.5:1), and side-by-side/
  stacked layout without horizontal overflow.
  Screenshots were inspected. Desktop uses click and keyboard; mobile uses
  keyboard activation because WebDriver mouse coordinates did not reliably
  follow emulated scrolling. Real touchscreen devices were not tested.
- The browser check uses synthetic sources and mocked generation; no live Gemini,
  CAPTCHA, or hosted-memory request was made. No HF/cloud deployment occurred.
  SQL is unchanged, so DB checks were not repeated for this phase.
- `docker build -t agenticrag-ux-phase1-check:local .` — production image built
  successfully with cached dependencies; existing Docker keyword-casing warning.
- `docker run --rm --network none --entrypoint python
  agenticrag-ux-phase1-check:local -m pip check` — no broken requirements.
  A network-disabled container with mock configuration also imported the runtime
  and built the revised Gradio UI successfully.
- `git diff --check` passed. Reviewed the application, source descriptors,
  adjusted existing tests, new regression/browser checks, and documentation.
- Commit policy: one meaningful commit per completed, verified phase, including
  this checklist.

### Phase 2 — Evidence-first answers

- **Items 10–15 implemented.** A synchronized per-turn collector captures only
  authorized retrieval results and assigns numeric source IDs. Source titles,
  excerpts, PDF page positions and original-PDF links are independent of model
  prose. Unknown numeric references cannot become valid citation links; model
  destinations on numeric references are replaced with registered-source actions.
- The web tool sends numbered passages rather than crop URLs/internal object
  keys. Images render only when a citation/card is selected. The CLI/default
  search-tool contract and the loopback-only legacy S3 workflow remain supported.
- Cards distinguish “Cited in this answer” from “Retrieved evidence · not cited.”
  Neither label implies factual verification. Retained answers use native,
  keyboard-operable disclosure groups. Selected previews include the source name,
  PDF file-page position, and a full-size image link.
- Evidence state retains at most 12 answers/120 passages, with at most 35 passages
  per turn and 2,000 characters per excerpt. Citation IDs are not reused while a
  conversation continues. Provider context is trimmed with evicted answer details;
  follow-up actions preserve remaining context and the selected execution options.
- Retained passages are reauthorized before display/selection. Source revocation
  during generation withholds the new prose as well as references; revoked tool
  results are removed before a subsequent follow-up reaches the provider. Ending,
  switching or clearing a session invalidates pending answers/previews and clears
  its evidence. Failed generation does not commit partial evidence/history.
- Web image URLs use an opaque `/showcase/citations/<hash>.png` route with current
  chunk/source authorization on every fetch and `Cache-Control: no-store`. Gradio
  file serving blocks the entire showcase-data directory. Internal file paths do
  not appear in the web evidence UI or the structured model-tool context.
- `PYTHONDONTWRITEBYTECODE=1 make test`: 96 passed, 39 dependency/fixture warnings.
  New coverage includes parallel ID assignment/deduplication, omitted/invalid
  citations, escaped content, missing geometry, retention/context limits, failed
  turns, source revocation, two-session/identity isolation, opaque image access,
  late private previews, historical selection, and follow-up settings/context.
- Real Chrome check (`python -m tests.browser_demo_ux --artifacts <temporary-directory>`)
  passed at 1360px desktop and 390px mobile-emulated widths. It exercises citation
  links and source-card buttons, actual synthetic PNG loading through opaque
  routes, earlier-answer selection after follow-ups, unchanged execution settings,
  clear/welcome behavior, report links, keyboard controls, layout and contrast.
  Browser inspection caught and fixed retained-card overlap and a JavaScript
  compatibility issue with Gradio 6.9. Mobile activation uses the keyboard;
  physical touchscreen devices and live provider/CAPTCHA flows are not verified.
- `make lint` and `git diff --check` passed. Production verification:
  `docker build -t agenticrag-ux-phase2-check:local .` passed (existing keyword-
  casing warning); `docker run --rm --network none --entrypoint python
  agenticrag-ux-phase2-check:local -m pip check` reported no broken requirements.
  A network-disabled container with mock configuration also built the evidence UI
  and validated citation formatting. No dependencies or SQL schema were changed;
  database checks were not repeated. No cloud deployment or live inference ran.
- Phase 2 is committed as `b08d545`.

### Phase 3 — Honest engineering visibility

- **Items 16–20 implemented.** The browser streams actual stage events from the
  existing agent: preparing authorized research, query optimization, generation,
  searching reports, and checking grounding when enabled. Gradio supplies its
  queue indicator before execution. No completion percentage or simulated timer
  is presented. The synchronous `submit` API remains available for integrations.
- `execution.py` captures request-local operational facts: observed stages,
  actual optimized sub-queries/fallback status, search calls/errors, agent
  generation calls, retrieved/cited counts and cumulative timings. Details are
  expandable and scoped to the current session. Router retries, token usage and
  monetary cost are not inferred; parallel timing overlap is explained.
- Grounding results are **Not run, Completed, Flagged, or Unavailable**. API
  failures, malformed JSON, missing fields and non-boolean verdicts no longer
  count as passes. The checker evaluates only this request's retrieved context,
  excluding prior turns and tool-error messages. Completed is not a guarantee of
  correctness. Unavailable and Flagged outcomes are visible in the answer/details.
- Query optimization validates one to three nonempty, bounded queries and
  records fallback to the original question. Raw provider exceptions are kept
  out of progress, public tool errors and execution details. Rate-limit guidance
  distinguishes potentially temporary rate limits from daily quota exhaustion.
- Session lifecycle guards also protect buffered stage events, execution details,
  and streamed final outputs. Clear/end/switch resets these details; a late
  private response cannot restore its sub-queries or answer in a newer session.
- “How this works” distinguishes historical FEMA-based metrics, the IPCC
  nine-combination retrieval smoke checks, and per-request automated judgements.
  It does not present Recall@1 as speed or historical scores as current-demo
  guarantees. A science-specific quality benchmark remains pending.
- Verification: `PYTHONDONTWRITEBYTECODE=1 make test` — **116 passed**, 40 existing
  dependency/fixture warnings; `make lint` — 0 configured issues;
  `git diff --check` — passed. New tests cover strict verdict schemas, service/
  JSON errors, decomposition fallback, real graph stages/counts, current-turn
  grounding context, disabled checks, empty responses, escaped diagnostics,
  retrieval failures, rate-limit recovery and private streaming invalidation.
- Browser command: `PYTHONDONTWRITEBYTECODE=1 python -m tests.browser_demo_ux
  --artifacts <temporary-directory>` — desktop (1360px) and mobile-emulated
  (390px) passed. The actual graph/optimizer/checker execute with synthetic
  provider responses and retrieval data. Checks include a visible generation
  stage, optimized queries and measured counts, disabled/unavailable grounding,
  citations, image routes, retained evidence, follow-ups, keyboard controls and
  layout. No live Gemini, CAPTCHA, hosted memory or physical touch-device check ran.
- `docker build -t agenticrag-ux-phase3-check:local .` — passed with cached
  dependencies and the existing keyword-casing warning. Network-disabled
  `python -m pip check` in that image reported no broken requirements; an
  additional container smoke check built the execution UI and confirmed an API
  failure produces an Unavailable verdict. SQL/dependencies were unchanged, so
  database checks were not repeated. No push or cloud deployment was performed.
- Phase 3 is committed as `689b1ae`.

### Phase 4 — Complete visitor-journey validation

- **Item 21 complete for local synthetic validation.** The expanded browser
  runner covers the full journey and has a `--memory-only` mode to avoid rerunning
  completed public checks while investigating private-session behavior.
- Desktop (1360px), mobile-emulated (390px) and independent-browser identity
  checks passed, including explicit opt-in, encrypted HttpOnly cookies, save/
  immediate panel update, reload, Forget, independent saved memories and returning
  to public mode without remote deletion. Public checks cover citations, earlier
  sources, follow-ups, unavailable/missing evidence, provider failure/retry,
  keyboard interaction, and actual graph execution details.
- Fixed confirmed Gradio 6.9 UI regressions found by these checks: loading
  indicators persisted across completed multi-component identity refreshes,
  and JS-only private-session actions did not reliably run their `.success()`
  refresh. Async JS actions now return their inputs to a backend refresh in the
  same event; errors still stop that refresh and server identity checks remain.
- Verification: **116 pytest tests passed**, lint reported 0 configured issues,
  **28 pgTAP checks passed** on disposable PostgreSQL 17.11, and the production
  image build/dependency check passed (existing Docker casing warning). The full
  browser journey passed after fixes. Earlier exploratory browser runs encountered
  Chrome disconnects; the final run used fresh renderer/profiles per viewport.
- [UX_VALIDATION.md](UX_VALIDATION.md) records the command sequence, coverage
  matrix, fixture boundaries, results and explicit pending hosted/live checks.
  No real provider/CAPTCHA/hosted-memory operation, touch-device test, push or
  deployment was performed. Local completion does not establish hosted readiness.
- **All four planned UX phases are locally implemented/validated. Next: hosted
  live-provider/CAPTCHA/memory smoke checks and corpus-specific quality evaluation.**
