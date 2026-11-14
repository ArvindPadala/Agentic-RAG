# Provider routing and request budgets

Implemented and verified locally on 13 November 2026.

## Defaults and configuration

| Variable | Default | Accepted range | Meaning |
|---|---:|---|---|
| `PROVIDER_TURN_TIMEOUT_SECONDS` | 90 | 0.1–300, finite | Shared monotonic scheduling budget for one agent turn |
| `PROVIDER_CALL_TIMEOUT_SECONDS` | 30 | 0.1–60, finite | Maximum SDK socket timeout for each dispatch |
| `PROVIDER_MAX_ATTEMPTS` | 12 | Integer 1–100 | Maximum SDK dispatch attempts across all stages and retries in the turn |
| `PROVIDER_MAX_ATTEMPTS_PER_CALL` | 3 | Integer 1–6 | Maximum dispatch attempts for one logical generation call |

Set these in `.env` locally or as host environment variables. Invalid settings
fail configuration validation; restart the application after changing them.
A call's effective timeout is capped by the
remaining turn budget and any shorter caller-supplied SDK timeout. The SDK
receives milliseconds and a corresponding `X-Server-Timeout` hint.

The turn starts inside `run_agent_turn`, before query decomposition. It includes
elapsed time spent in decomposition, generation, tools, retries and grounding;
Gradio queue waiting, readiness checks and private-session loading precede it.
Standalone router calls, such as memory extraction outside the agent, receive
their own finite budget. An inner scope cannot reset an existing turn's limits.

### Deadline semantics

This is a **cooperative scheduling/admission deadline**, not hard cancellation:

- No further SDK dispatch may start after expiry. A late successful response is
  rejected before it can enter accepted history or grounding.
- The graph checks the budget at stage/tool boundaries, with context propagation
  into tool workers. A tool returning after expiry cannot trigger another search
  or generation cycle.
- HTTPX's synchronous timeout is an inactivity/phase limit, not a total stopwatch
  for connect + write + read + decode. An already-running SDK request or retrieval
  function cannot be forcibly interrupted here, so total wall time can exceed
  the scheduling budget before control returns and the result is rejected.
- No background timeout threads or supposedly cancelled SDK calls are left
  running. Strict end-to-end wall-clock cancellation would require cancellable
  I/O/worker orchestration and remains outside this increment.

## Routing and retry policy

- SDK clients and policy are shared; route cursors and cooldowns are **local to
  one request**, keyed by router and requested model. A working route is retained
  for later calls to that model within the turn; independent turns start fresh.
- Routes try the requested model's configured keys before subsequent fallback
  models. Duplicate keys/model entries are removed. Defaults are
  `models/gemini-3.6-flash` and `models/gemini-3.5-flash-lite`; a constructor
  `models=[...]` explicitly supplies the fallback list. A different explicitly
  requested stage model starts with that model, not another stage's sticky route.
- **Retryable HTTP statuses:** 408, 429, 500, 502, 503 and 504. Connection errors
  and connection timeouts may retry. Other HTTP statuses, including 400/401/403/
  404/422, are not retried. In particular, 404 is a configuration/model failure,
  not a reason to silently mask the error with another model.
- Authentication/permission failures (401/403) stop further dispatches in the
  turn. Structured `google.rpc.QuotaFailure` evidence of a daily or zero quota
  also stops the turn without rotating around an exhausted quota.
- Other transient failures use bounded exponential jitter. Valid `Retry-After`
  seconds/HTTP dates and `google.rpc.RetryInfo.retryDelay` minimums take priority
  when longer. A minimum wait that cannot fit the remaining budget fails without
  sleeping or being bypassed by an optional-stage fallback.
- Read/write/pool timeouts are **not automatically retried**: a request may have
  already generated a billable result. There is no inference that retries are
  free, that two keys have independent quotas, or that key rotation removes limits.
- SDK HTTP retries are explicitly disabled at both client and request level.
  SDK automatic function calling is disabled because LangGraph owns tools. This
  keeps router dispatch accounting from hiding additional SDK-generated calls.
- SDK errors are reduced to a safe type/status. Logs, diagnostics and exception
  messages do not serialize raw provider bodies, request URLs or credentials.

Limits are per turn, not a process-wide rate limiter or fair queue. Many requests
can still exhaust the project's quota; distributed/global admission control and
load testing remain separate work. The follow-up below validates local default
model availability; repeat it for deployment-specific credentials and dates.

## Outcomes and diagnostics

The web handler preserves prior completed context on required-call timeout or
attempt exhaustion. It reports `timed_out` or `budget_exhausted`, rather than
publishing a completed answer. Evaluation rows preserve those specific outcomes
and stop the remaining cases after saving a non-success result.

An already complete answer can still be returned with grounding **Unavailable**
if the optional judge cannot reserve another attempt while time remains. If the
overall deadline expires during grounding, the answer is withheld and history
is rolled back. Decomposition may fall back to the original query; it cannot
reset the shared deadline, attempt cap, auth failure or retry cooldown.

Execution details now record actual router SDK dispatch attempts across all
stages, total measured retry waiting, attempted model IDs and configured limits.
Logical agent-generation counts remain separate. A synthetic client that bypasses
the router records no real router dispatches. No token/cost totals are inferred.

## Verification actually executed

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_ADDOPTS='-p no:cacheprovider' make test
PYTHONDONTWRITEBYTECODE=1 make lint
PYTHONDONTWRITEBYTECODE=1 python -m tests.browser_demo_ux \
  --artifacts /var/folders/5n/2qvbhdkn6jx2995yk4_5jv680000gn/T/opencode/provider-budget-browser
docker build -t agenticrag-provider-budget-check:local .
docker run --rm --network none --entrypoint python \
  agenticrag-provider-budget-check:local -m pip check
docker run --rm --network none -e GEMINI_API_KEY=provider-smoke-placeholder \
  -e SUPABASE_ENABLED=false -e PYTHON_DOTENV_DISABLED=1 --entrypoint python \
  agenticrag-provider-budget-check:local -c \
  'from config import settings; from agent import create_gemini_router; router = create_gemini_router(settings.GEMINI_API_KEYS); assert router.policy.turn_timeout_seconds == 90; assert router.policy.max_attempts == 12; import app; print("Runtime imports and provider policy smoke passed; no provider call made")'
```

- Initial budget increment: **197 passed**, 43 dependency/fixture warnings.
- `make lint`: **0 configured issues**; whitespace checks passed.
- Final focused check after the logging/documentation refinement:
  `PYTHONDONTWRITEBYTECODE=1 PYTEST_ADDOPTS='-p no:cacheprovider' python -m pytest tests/test_llm_router.py tests/test_deployment_privacy.py -q`:
  **44 passed**, 6 warnings. The source image was refreshed with cached dependency layers.
- Real Chrome desktop/mobile-emulated public flows passed with synthetic
  provider/data. These are not live provider-load or hosted HTTPS checks.
- The initial `tests/test_llm_router.py` suite had 42 deterministic cases, including concurrent
  multi-call requests, shared real-graph stage/tool budgets, late-result rejection,
  limits/cooldowns, error redaction and permanent/transient failures. An actual
  Google SDK + HTTPX mock transport verifies per-request timeout delivery and that
  caller-requested SDK retries cannot multiply router attempts.
- Production image built successfully on the local Linux ARM64 Docker platform;
  the existing Docker keyword-casing warning remains. Network-disabled `pip check`
  found no broken requirements. A network-disabled smoke imported config/agent/UI
  and constructed a router with a placeholder key; it made no provider call.
- Installed host SDK/HTTPX versions: `google-genai==1.66.0`, `httpx==0.28.1`.
  HTTPX is now a declared direct dependency; the SDK minimum requires the options
  used here. No host packages were installed or upgraded by this work.

### Targeted live checks

Only the unfinished benchmark case and one positive sample were selected. Both
used the approved IPCC index, real provider and grounding, with new output files:

```bash
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m eval.evaluate_showcase --live --guardrail \
  --case unsupported-current-price --output eval/results/ipcc_provider_budget_check.json
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m eval.evaluate_showcase --live --guardrail \
  --case technology-costs --output eval/results/ipcc_provider_budget_positive.json
```

| Case | Outcome | Router dispatches | Retry waiting | Agent-turn time |
|---|---|---:|---:|---:|
| `unsupported-current-price` | Completed; correctly declines to invent a current local price | 4 | 0 s | 29.07 s |
| `technology-costs` | Completed; solar 85%, wind 55%, batteries 85%, with a valid WGIII citation | 3 | 0 s | 3.93 s |

Both final agent candidates reported `STOP`; both grounding checks completed.
Models attempted were `models/gemini-3.6-flash` for generation and
`models/gemini-3.5-flash-lite` for grounding. There were no transient errors in
these two live checks, so real-world retry recovery is not demonstrated by them;
the SDK mock-transport and deterministic failures cover that behavior.

These observations do not prove that the policy caused a latency improvement or
eliminated quota failures: quota state and timing differ from the earlier run.
Original baseline reports/scores were preserved. Detailed quality context is in
[eval/IPCC_EVALUATION.md](eval/IPCC_EVALUATION.md).

During the initial budget increment, no HF push/deployment, live private-memory/
Auth mutation, SQL change, physical touch-device test, AMD64 image run, process-
load test or full quality rerun occurred. The later concurrency checks follow below.
The base-image tag and transitive dependencies remain unpinned; a successful local
build/import smoke is not a reproducibility or hosted-readiness guarantee.

## Availability and local concurrency follow-up

### Verified P1 defect and fix

The formerly configured fallback `models/gemini-3.6-flash-lite` returned HTTP 404
from metadata for **both local credentials**, and then from generation on one
targeted case. The generation failure consumed exactly one router dispatch,
confirming that permanent model/configuration errors were not retried.

The available `models/gemini-3.5-flash-lite` completed the same IPCC technology-cost
case with actual tool calling, a correct cited answer, normal `STOP`, and a
completed grounding check: three dispatches, no retries, approximately 2.75 s
of agent-turn processing. The default fallback was changed to this verified
model. A new regression saturates both primary-key routes with synthetic 503s
and checks that the next dispatch selects the new default fallback.

After the change, all **six metadata checks** passed: the three distinct default
IDs (generation primary, fallback/planner/judge, and memory extractor), each under
two credential slots. No credential values or raw error bodies are in the report.
Metadata availability still does not prove quota, quality or every model capability.
The functional generation sample used one credential; these are not live forced-
outage tests across all credentials and roles.

`eval/check_provider_release.py models --live` performs metadata reads only, with
a ten-second-or-shorter SDK timeout and one SDK attempt per read. It skips remaining
IDs for a credential after auth/throttle/overload failures. It does not generate.
The functional checks use the existing showcase evaluator separately.

```bash
# Initial metadata: expected failure identifying the bad fallback; preserved.
PYTHONDONTWRITEBYTECODE=1 python -m eval.check_provider_release models --live
# Confirmed generation 404: expected failure, one dispatch; preserved.
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m eval.evaluate_showcase --live --case technology-costs \
  --model models/gemini-3.6-flash-lite --output eval/results/unavailable_fallback_generation.json
# Functional candidate check before replacing the default: completed.
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m eval.evaluate_showcase --live --guardrail --case technology-costs \
  --model models/gemini-3.5-flash-lite --output eval/results/replacement_fallback_generation.json
# Updated default roles: six metadata reads passed.
PYTHONDONTWRITEBYTECODE=1 python -m eval.check_provider_release models --live \
  --output eval/results/provider_models_after_fix.json
```

### Measured local burst with synthetic provider transport

The same tool has a `load` mode. It disables dotenv/private sessions/tracing,
uses synthetic SDK keys and an HTTPX mock transport, and runs the **actual SDK,
router, agent graph, public IPCC retrieval/reranking, evidence and grounding
control flow**. Generated answers and judge verdicts are synthetic. No real
generation requests or personal memory reads occur in these load experiments.

Each run submits eight requests into a local executor, with 50 ms synthetic
provider latency per dispatch. Even-numbered cases receive one or two initial
429s. Passing requires captured/cited authorized evidence, its own request marker,
successful judge control flow, and independently consistent SDK dispatch counts.
Two injected failures also require the synthetic fallback model; other cases
must not unexpectedly switch to it.

| Warm run | Workers | Injected 429s | Successful requests | SDK dispatches | Burst seconds | Service p95 | Queue p95 | Total p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `provider_concurrency_one_warm.json` | 1 | 4 | 8/8 | 28 | 8.28 | 1.54 s | 7.76 s | 8.27 s |
| `provider_concurrency_two_warm.json` | 2 | 4 | 8/8 | 28 | 4.26 | 1.64 s | 3.35 s | 4.26 s |
| `provider_concurrency_failover_warm.json` | 2 | 8 | 8/8 | 32 | 7.02 | 3.01 s | 5.73 s | 7.02 s |

All contexts stayed separate and all per-request dispatch ledgers matched.
Peak simultaneous mock-provider calls were one, two and two respectively. These
are descriptive nearest-rank percentiles over eight requests, not service SLOs.
The shared host was not isolated, other verification was running, and production
retry jitter was not fixed. Do not infer a causal scaling factor or Gemini/HF
capacity from these numbers. The local executor models a burst at the agent
layer, not Gradio's actual HTTP/SSE queue, frontend behavior or anonymous-user abuse.

Graph imports, model loading and an initial retrieval warmup occur before timing
the burst. Worker queue time stops at function entry, not after lazy imports.
Initial non-`_warm` runs passed operational invariants but included a lazy-import
timing issue; they were retained for diagnosis and excluded from the table. The
three corrected timing experiments were rerun serially with new output names.
Process-lifetime peak RSS was approximately 1.11–1.17 GB (decimal), including setup;
it is not a per-request memory allocation or a cold-start/concurrent-init guarantee.

```bash
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m eval.check_provider_release load --requests 8 --concurrency 1 --fail-first 1 \
  --output eval/results/provider_concurrency_one_warm.json
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m eval.check_provider_release load --requests 8 --concurrency 2 --fail-first 1 \
  --output eval/results/provider_concurrency_two_warm.json
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m eval.check_provider_release load --requests 8 --concurrency 2 --fail-first 2 \
  --output eval/results/provider_concurrency_failover_warm.json
```

Request count is bounded to 1–32, worker count to 1–4, synthetic latency to
0–2000 ms, and injected initial failures to 0–2. Reports save after every
completed request under ignored `eval/results/`. `--live` exists only on the
metadata subcommand. First showcase/model provisioning can download public
assets if missing; the executed load runs reused cached, checksum-pinned assets.

Current follow-up verification: **204 pytest tests passed**, 43 warnings;
configured lint **0 issues**. This includes the new default-fallback regression
and six release-probe tests (metadata without generation, safe failure/skips,
synthetic transport behavior, mixed-context rejection, failure-inclusive latency
summaries, and actual SDK/graph/evidence worker integration). Exact regression
commands remain `PYTHONDONTWRITEBYTECODE=1 PYTEST_ADDOPTS='-p no:cacheprovider' make test`
and `PYTHONDONTWRITEBYTECODE=1 make lint`.

Final focused checks also passed: 49 router/release-probe cases with six warnings.
The production image was refreshed after changing the default, and this
network-disabled import/default check passed (no provider request):

```bash
docker build -t agenticrag-provider-budget-check:local .
docker run --rm --network none -e GEMINI_API_KEY=model-check-placeholder \
  -e SUPABASE_ENABLED=false -e PYTHON_DOTENV_DISABLED=1 --entrypoint python \
  agenticrag-provider-budget-check:local -c \
  'from llm_router import GeminiRouter; router = GeminiRouter(["model-check-placeholder"]); assert router.fallback_models == ("models/gemini-3.6-flash", "models/gemini-3.5-flash-lite"); import app; print("Verified fallback defaults and runtime imports; no provider call made")'
```

The next release actions are to commit the reliability work, resolve the separate
HF artifact inventory, and validate the intended revision on its HTTPS host.
App queue limits/global admission, real provider concurrency/quota behavior,
cold starts, and repeatability on the hosted architecture remain unmeasured.
