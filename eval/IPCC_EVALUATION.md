# Current IPCC showcase evaluation

Executed locally: 25 October 2026 (26 October UTC in report timestamps).

## Outcome

The current IPCC pipeline has a reproducible **12-case development benchmark**.
Source validation and all 30 direct-retrieval/preview checks completed. Live agent
evaluation saved **11 completed answers**; the remaining case failed with provider
HTTP **429** after a targeted continuation. This is a measured starting baseline,
not a complete quality pass or a hosted-release approval.

| Direct search mode | Answerable questions scored | Annotated group coverage @5 | All annotated groups found @5 | MRR of first annotated hit @5 |
|---|---:|---:|---:|---:|
| Vector | 10 | 70% | 7/10 | 0.3783 |
| Hybrid RRF | 10 | 80% | 8/10 | 0.5333 |
| Hybrid + CPU reranker | 10 | 80% | 8/10 | 0.6500 |

Every mode returned results and rendered a preview of its first result on all
ten answerable cases: **30/30 previews**. A rendered preview proves that the
retrieved source can be displayed, not that it supports the question.

### What these metrics mean

- A case contains one or more **source-qualified, exact-page quote witnesses**.
  Witnesses resolve to immutable chunk IDs before running retrieval. Cross-report
  cases require independent witnesses from two report identities.
- Group coverage is the fraction of annotated evidence groups represented in
  the top five passages. All-groups hit rate requires every group in the case.
  MRR uses the rank of the first annotated hit. Errors remain zero-scored in the
  denominator; empty/unrun mode aggregates are null, not passing zeros.
- The quotes are a small, non-exhaustive relevance annotation. Equivalent valid
  passages can miss this exact-witness score. These are **not full relevance
  recall, comprehensive answer accuracy, or population-level estimates**.
- The two unsupported cases have no retrieval score. Returning related climate
  passages neither establishes an exact city/day prediction nor a current price.
- The benchmark is an agent-authored, source-checked development set. It needs
  independent scientific review and a larger held-out set before quality claims.

## Corpus and provenance

- Three approved IPCC AR6 SPMs: WGI 2021, WGIII 2022, Synthesis 2023.
- **1,711 chunks**, collection `showcase_c2601d139b4fc129f062c9f0`, parser
  `pymupdf-blocks-v1`. Existing curated provisioning verifies exact expected IDs
  and publishes only that corpus. Legacy `chroma_db/` and personal memory are not
  used by this evaluator.
- Source SHA-256 pins are required to match `showcase_sources.json`. Benchmark
  file: `eval/ipcc_benchmark.json`; SHA-256 for these runs:
  `3a587b352db74faa357ab12874a7b55c852c418a3d79189e470d1dad88b5669f`.
- Baseline Git commit: `8bfeb92a283516564afc74c78da3e7ddbf48e1ee`.
  Evaluation code was uncommitted; later reports include implementation file
  hashes so that the baseline commit is not misrepresented as containing it.
- Python 3.12.7; Chroma 1.5.5; sentence-transformers 5.2.3; PyMuPDF 1.27.2.
  Chroma default MiniLM embeddings and `ms-marco-MiniLM-L-6-v2` CPU reranking.
  Model weights are not revision-locked by this new harness.
- Retrieval uses the existing public collection wrapper and existing search
  functions. Live evaluation uses the real LangGraph agent, search tool,
  evidence collector/authorization check, decomposition and grounding checker.
  Each case has fresh router/history/evidence, no persistent memory or S3.
- Requested live model: `models/gemini-3.6-flash`, fixed for comparability.
  Key rotation/retries use `GeminiRouter`; its model list is limited to the
  requested model, rather than silently mixing fallback model results. SDK
  returned model versions, token totals and router retry totals were not captured.

## Live results and answer review observations

The initial smoke question (`technology-costs`, no grounding) succeeded. The
subsequent batch enabled grounding for every case and decomposition for the two
cross-report cases, using reranked search for agent tool calls.

| Case | Saved live result | Review observation |
|---|---|---|
| `human-influence` | Completed | Main unequivocal-attribution conclusion present; expanded into five searches and a long answer |
| `observed-warming` | Completed | Recovered relevant SYR evidence through three searches; gave observed 1.09°C/approximately 1.1°C and principal human cause |
| `technology-costs` | Completed | Correct source-period reductions: solar 85%, wind 55%, batteries 85% |
| `demand-side` | Completed | 40–70% by 2050 and distributional caveat present |
| `pathway-reductions` | Completed | 43% versus 27%, probability/overshoot assumptions and ranges present; includes an unparsed grouped reference |
| `observed-sea-level` | Completed | Correct 0.20 m increase and historical rate/range comparison |
| `hot-extremes` | Completed | Preserves virtually certain/high-confidence qualifications |
| `net-zero-requirement` | Completed | Separates net-zero CO2 from net-zero GHG and cumulative emissions |
| `cross-report-warming` | Completed mechanically; incomplete answer | Cites both report identities and distinguishes periods/quantities, but ends mid-sentence during its explanation |
| `cross-report-costs` | Completed | Compares both reports' percentages/periods and rejects inference of current installed prices |
| `unsupported-city-date` | Completed | Refuses an exact Hyderabad/day temperature; its grouped references are not parsed as clickable evidence IDs |
| `unsupported-current-price` | Failed | Targeted continuation exhausted router attempts with HTTP 429; no final answer or grounding result |

All 11 completed answers had grounding status **completed**. This is the automated
judge's result, not an independent factual/completeness verdict. The incomplete
cross-report answer demonstrates why those must remain separate. These review
observations were made by the coding assistant against benchmark guidance;
`human_review` remains `pending` in the artifacts for independent review.

The ten completed answerable cases all captured and cited at least one annotated
passage. Nine captured every annotated group; `cross-report-warming` matched one
of two witnesses, while also citing a valid alternative SYR passage from the
same page. Mean annotated coverage across those answerable agent turns was 95%,
not an answer-correctness score. It is different from single-query retrieval.

Ten completed answers contained recognized individual numeric references, all
resolving to captured evidence; no invalid **recognized** IDs were observed.
This excludes unparsed groups such as `[1, 4]`. It does not measure whether every
claim is cited or entailed. At baseline, the evaluator flagged these groups as
unparsed. The grouped-citation follow-up below changes the parser and future
evaluation accounting; the original measurements remain unchanged.

### Interrupted batch and continuation

The batch reached the terminal's **240-second timeout** while processing its last
case. It had atomically saved 11 completed cases. Its report retains
`run_status: running`, accurately indicating interruption; it was not rewritten
to appear successful. Only the unfinished case was attempted again in a separate
report. That continuation saved `ClientError`, code 429, after about 48.73 seconds,
with grounding `not_run`. No additional live retries were attempted afterward.

The completed case times ranged from about **5.45 to 48.10 seconds**. These are
single-run observations including some quota recovery, not load-test percentiles.
Cold model loads, provider retries and variable search counts make an aggregate
CPU/GPU or capacity claim inappropriate.

## Verified improvements to prioritize

1. **P1 — grouped citation rendering.** `evidence.py:15,92-113` recognizes single
   numeric IDs only. Saved `pathway-reductions` and `unsupported-city-date`
   answers use grouped references, so source navigation can be missing despite
   apparently cited prose. Normalize/validate each group member without accepting
   model-provided destinations, and regression-test unknown members and numeric
   scientific ranges. The application formatter was subsequently fixed in the
   grouped-citation follow-up below.
2. **P1 — answer completeness and quota/deadline handling.** The saved
   `cross-report-warming` answer stops at “from natural”, then gets a completed
   grounding footer. `agent.py:289-294,350-355,405-448` limits output but does not
   classify provider finish reasons before accepting text/grounding. Truncation
   is observed; its cause was not captured. Record finish reasons, surface
   incomplete responses and bound unnecessary searches/overall request duration.
   The 429 failure and interrupted batch also support fixing provider budgets
   and adding resumable orchestration before unattended evaluations at scale.
3. **P2 — initial-query retrieval coverage.** Both `observed-warming` and
   `cross-report-warming` missed their annotated witnesses in all three modes;
   vector search also missed `net-zero-requirement`. Agent refinements recovered
   useful evidence at additional latency/call cost. Investigate query/source
   selection and passage ranking using this baseline, then verify on new held-out
   questions rather than tuning the benchmark prompts after seeing results.

## Commands actually executed

From the repository root:

```bash
# Inspect public passages while preparing annotations; no generation/indexing.
PYTHONDONTWRITEBYTECODE=1 python -m eval.evaluate_showcase \
  --find 'unit costs' --find 'unequivocal' --find '40–70%' --find '1.1°C' --find '43%'
PYTHONDONTWRITEBYTECODE=1 python -m eval.evaluate_showcase \
  --find '0.20' --find 'cumulative CO2' --find 'hot extremes' --find '1.07'

# All 12 case definitions/source-qualified quote witnesses validated.
PYTHONDONTWRITEBYTECODE=1 python -m eval.evaluate_showcase --validate-only

# LLM-free retrieval run with cached model weights.
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m eval.evaluate_showcase

# Successful real-provider smoke.
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m eval.evaluate_showcase --live --case technology-costs \
  --output eval/results/ipcc_live_smoke.json

# Saved 11 live cases; terminal interrupted at 240 seconds on the final case.
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m eval.evaluate_showcase --live --guardrail \
  --output eval/results/ipcc_full_pipeline.json

# Only the unfinished case; saved failure with HTTP 429.
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m eval.evaluate_showcase --live --guardrail \
  --case unsupported-current-price --output eval/results/ipcc_remaining_case.json

# Review saved answers without regenerating them.
PYTHONDONTWRITEBYTECODE=1 python -m eval.evaluate_showcase \
  --review-report eval/results/ipcc_full_pipeline.json
PYTHONDONTWRITEBYTECODE=1 python -m eval.evaluate_showcase \
  --review-report eval/results/ipcc_remaining_case.json

# Final regression checks: 123 passed / 40 warnings; lint 0 configured issues.
PYTHONDONTWRITEBYTECODE=1 PYTEST_ADDOPTS='-p no:cacheprovider' make test
PYTHONDONTWRITEBYTECODE=1 make lint
```

Use `--case` repeatedly to run selected IDs. A live operational error stops the
remaining cases after saving a sanitized failure. The runner saves after each
case, but it does not automatically resume or impose a total wall-clock deadline.
Before resuming, inspect the saved results and select only unfinished cases with
a new output filename. Do not overwrite the original evidence of an interruption.

## Files and scope

- `eval/ipcc_benchmark.json`: versioned questions, reference quotes/pages,
  review guidance, source pins and parser identity.
- `eval/evaluate_showcase.py`: source validation, direct search comparison,
  optional live agent evaluation, atomic JSON reports and saved-answer review.
- `tests/test_showcase_evaluation.py`: seven regressions covering source/page
  binding, scoring, actual graph/evidence integration, failure redaction and
  stop-on-failure persistence.
- `eval/results/`: generated local reports, ignored by Git and excluded from
  HF/Docker packaging. Keep these files locally when comparing future changes.

The evaluator provisions/reuses the approved showcase; first provisioning may
download public PDFs and model weights. Offline here means no generative-provider
calls, not a network sandbox. Live runs load operator credentials through existing
configuration but never print them; public tracing is disabled and no private
memory/Auth/AWS operation is invoked. There were no dependency updates, commits,
HF uploads, live browser/Auth tests, Docker builds or SQL changes in this work.

## Follow-up: grouped citation parsing fixed

The UI formatter now expands `[1, 4]` and `[[1, 4]]` into separate captured-source
links. It validates each member independently, retains valid members beside
unknown ones, deduplicates members within a group, and discards model-supplied
link destinations. Existing individual and already-rendered links remain usable.

Decimal values, percentages, signed values, zero-based lists, dash ranges,
escaped references and Markdown image labels are preserved. Bare comma-separated
positive integer lists are reserved for citation syntax: their meaning cannot
be distinguished reliably from arbitrary integer data by syntax alone.

The evaluator shares `citation_numbers()` with the renderer, so groups contribute
to citation metrics and cited-evidence coverage. New reports retain both raw and
rendered answers; residual grouped-reference diagnostics inspect the rendered
answer instead of flagging every raw group as broken.

The two affected saved IPCC answers were replayed without generation:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m eval.evaluate_showcase \
  --review-report eval/results/ipcc_full_pipeline.json --recheck-citations \
  --case pathway-reductions --case unsupported-city-date
```

- `pathway-reductions`: `[1, 5]` becomes two valid source links; numerical ranges
  such as `[34–60%]` remain unchanged. The same IDs were already cited elsewhere,
  so the cited-ID set stays `{1, 2, 5}`.
- `unsupported-city-date`: `[1, 4]` and `[1, 6]` become valid source links, with
  newly recognized cited IDs `{1, 4, 6}` and no unknown IDs.

This is **formatting replay against saved evidence**, not a recheck of current
source authorization or a new grounding/answer-quality verdict. The saved
baseline, live failure and original retrieval measurements were not modified.

Verification after the fix:

- `PYTHONDONTWRITEBYTECODE=1 PYTEST_ADDOPTS='-p no:cacheprovider' make test`:
  **138 passed**, 41 dependency/fixture warnings. Coverage includes the real web
  callback's cited state, actual agent/evaluator flow, numerical non-citations,
  model-controlled destinations and a non-mutating saved-report replay.
- `PYTHONDONTWRITEBYTECODE=1 make lint`: **0 configured issues**.
- `git diff --check`: passed.

No new provider calls, browser runs, SQL checks, image builds or deployments were
needed for this formatting follow-up. Incomplete-response and quota/deadline
handling remain separate follow-up work.

## Follow-up: provider completion handling

Implemented and verified locally on 2 November 2026.

The agent now accepts only an explicit `STOP` finish reason from its chosen
provider candidate before using text or executing tools. `MAX_TOKENS`, other
non-normal reasons and missing/unknown metadata raise a safe
`IncompleteResponseError`. Partial text is neither displayed nor attached to
the exception or saved as an accepted answer. Empty/whitespace or thought-only
responses also fail without a completed grounding verdict.

Failed and interrupted graph turns restore prior conversation history, including
CLI Ctrl-C before memory-on-exit. The web request reports `incomplete` with a
retry/narrower-question message and no new evidence turn. There is no automatic
regeneration, increased output budget or extra judge call for incomplete answers.

The same completion requirement applies to structured grounding/decomposition
responses, even if their JSON happens to parse. An incomplete judge response
becomes `unavailable`; incomplete decomposition falls back to the original query.
New execution snapshots include `finish_reason` for the final attempted agent
call, and evaluator cases report `incomplete` instead of `completed`. The batch
stops on that non-success outcome after saving diagnostics. Unknown provider
metadata is normalized rather than exposing arbitrary strings.

These finish reasons establish transport/provider termination, **not semantic
completeness or factual accuracy**. The original cross-report answer had no saved
finish reason, so this follow-up does not retroactively prove why it ended
mid-sentence. Historical reports and measured retrieval scores remain intact.

Checks actually executed:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_ADDOPTS='-p no:cacheprovider' make test
PYTHONDONTWRITEBYTECODE=1 make lint
PYTHONDONTWRITEBYTECODE=1 python -m tests.browser_demo_ux \
  --artifacts /var/folders/5n/2qvbhdkn6jx2995yk4_5jv680000gn/T/opencode/incomplete-response-browser
git diff --check
```

- Final full suite: **153 passed**, 43 dependency/fixture warnings.
- Configured lint: **0 issues**; diff whitespace check passed.
- Real Chrome with synthetic provider/data: desktop and mobile-emulated public
  research flows passed, including graph stages, decomposition, unavailable
  grounding, citations/previews, follow-ups and clear. This smoke run verifies
  normal `STOP` compatibility; targeted graph/web-handler regressions exercise
  incomplete/blocked/missing finish reasons and rollback.
- Synthetic SDK candidates explicitly declare finish reasons. Regressions cover
  aborted tool calls, a truncated final answer after a successful tool cycle,
  parseable-but-incomplete structured JSON, empty/hidden-only answers, ordinary
  provider failure and operator interruption, and incomplete evaluator rows.

No new live Gemini/Auth calls, physical-device checks, SQL checks, container
builds, dependency changes or deployments were performed. Provider deadlines,
retry classification/routing, pacing and the remaining live quota failure still
need the next reliability increment before another targeted live evaluation.
