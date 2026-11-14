"""Read-only live model metadata or a local IPCC-agent concurrency experiment.

Metadata uses configured credentials only with `models --live` and never calls
generation. `load` replaces the provider transport with synthetic responses and
uses only the approved showcase index; it never loads operator credentials.
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import re
from threading import RLock
import time
from unittest.mock import patch

from eval.evaluate_showcase import error_details


def configured_models(router):
    from agent import run_agent_turn
    from live_guardrail import check_faithfulness
    from query_optimizer import decompose_query

    roles = {}
    for function, role in ((run_agent_turn, "generation"), (decompose_query, "decomposition"),
                           (check_faithfulness, "grounding")):
        model = inspect.signature(function).parameters["model"].default
        roles.setdefault(model, []).append(role)
    for index, model in enumerate(router.fallback_models):
        roles.setdefault(model, []).append("router_primary" if index == 0 else "router_fallback")
    # The memory extractor currently selects this literal model in gemini_helpers.
    roles.setdefault("models/gemini-3.5-flash", []).append("memory_extraction")
    return roles


def probe_models(clients, roles, timeout_ms, save_row):
    from google.genai import types

    for slot, client in enumerate(clients, 1):
        blocked = None
        for model, uses in roles.items():
            row = {"credential_slot": slot, "requested_model": model, "roles": uses,
                   "status": "skipped" if blocked else "failed"}
            if blocked:
                row["blocked_by_code"] = blocked
                save_row(row)
                continue
            started = time.perf_counter()
            try:
                metadata = client.models.get(model=model, config=types.GetModelConfig(http_options=types.HttpOptions(
                    timeout=timeout_ms, retry_options=types.HttpRetryOptions(attempts=1))))
                supports_generation = "generateContent" in (metadata.supported_actions or [])
                row.update(status="metadata_available" if supports_generation else "metadata_inconclusive",
                           supports_generate_content=supports_generation)
            except Exception as error:
                row["error"] = error_details(error)
                code = row["error"]["code"]
                if code in (401, 403, 429, 503):
                    blocked = code
            row["seconds"] = time.perf_counter() - started
            save_row(row)


class SyntheticTransport:
    """Local SDK transport only. Includes deterministic 429 injection by case ID."""

    def __init__(self, latency_seconds, fail_first):
        self.latency_seconds = latency_seconds
        self.fail_first = fail_first
        self.events = []
        self.failures = {}
        self.active = 0
        self.peak_active = 0
        self._lock = RLock()

    def handle(self, request):
        import httpx

        body = json.loads(request.content)
        ids = set(re.findall(r"loadcase-\d{3}", json.dumps(body["contents"])))
        if len(ids) != 1:
            raise ValueError("Synthetic request marker is missing or mixed")
        request_id = next(iter(ids))
        instruction = json.dumps(body.get("systemInstruction", {}))
        judge = "strict faithfulness evaluator" in instruction
        tool_results = [part["functionResponse"]["response"]["result"]
                        for content in body["contents"] for part in content.get("parts", [])
                        if "functionResponse" in part]
        phase = "grounding" if judge else "answer" if tool_results else "tool_plan"
        with self._lock:
            self.active += 1
            self.peak_active = max(self.peak_active, self.active)
            failures = self.failures.get(request_id, 0)
            fail = phase == "tool_plan" and int(request_id[-3:]) % 2 == 0 and failures < self.fail_first
            if fail:
                self.failures[request_id] = failures + 1
            self.events.append({"request_id": request_id, "phase": phase, "status": 429 if fail else 200})
        try:
            time.sleep(self.latency_seconds)
            if fail:
                return httpx.Response(429, headers={"Retry-After": "0.05"}, json={"error": {
                    "code": 429, "status": "RESOURCE_EXHAUSTED", "message": "Synthetic test throttle"}})
            if judge:
                parts = [{"text": json.dumps({"is_faithful": True,
                           "reason": f"Synthetic judge result for {request_id}; not a quality assessment."})}]
            elif tool_results:
                match = re.search(r"Evidence \[(\d+)\]", " ".join(tool_results))
                if not match:
                    raise ValueError("Synthetic scenario did not receive captured evidence")
                parts = [{"text": f"Synthetic load response {request_id} [{match.group(1)}]. Not a scientific answer."}]
            else:
                parts = [{"functionCall": {"name": "search_knowledge_base", "args": {
                    "query": "2010 2019 solar wind lithium-ion battery unit costs"}}}]
            return httpx.Response(200, json={"candidates": [{"finishReason": "STOP", "content": {
                "role": "model", "parts": parts}}]})
        finally:
            with self._lock:
                self.active -= 1


def percentile(values, proportion):
    """Nearest-rank percentile; small samples are descriptive, not SLO estimates."""
    values = sorted(values)
    return values[max(0, math.ceil(len(values) * proportion) - 1)] if values else None


def summarize_load(rows, seconds):
    completed = [row for row in rows if row["status"] == "completed"]
    return {"requests_observed": len(rows), "completed": len(completed), "failed": len(rows) - len(completed),
            "experiment_seconds": seconds, "completed_requests_per_second": len(completed) / seconds if seconds else None,
            "service_p50_seconds": percentile([r["service_seconds"] for r in rows], 0.5),
            "service_p95_seconds": percentile([r["service_seconds"] for r in rows], 0.95),
            "queue_p95_seconds": percentile([r["queue_seconds"] for r in rows], 0.95),
            "total_p95_seconds": percentile([r["total_seconds"] for r in rows], 0.95)}


def load_request(request_id, submitted, router, collection, mode):
    started = time.perf_counter()  # Worker entry, before any lazy module import.
    from agent import build_agent_config, build_search_tool, run_agent_turn
    from access_control import RuntimePolicy
    from evidence import EvidenceCollector, authorized_passages, citation_numbers, format_citations
    from execution import ExecutionTrace

    trace, collector = ExecutionTrace(mode), EvidenceCollector()
    row = {"request_id": request_id, "status": "failed", "queue_seconds": started - submitted}
    try:
        policy = RuntimePolicy("public_demo")
        search, tool = build_search_tool(collection, router, None, None, use_hybrid=mode != "vector",
                                        use_reranker=mode == "reranked", evidence_collector=collector,
                                        execution=trace, policy=policy)
        answer = run_agent_turn(f"Request {request_id}: What were historical renewable energy cost reductions?", [],
                                router, build_agent_config(tool, {}, structured_citations=True),
                                {"search_knowledge_base": search}, model="models/synthetic-primary",
                                use_guardrail=True, execution=trace)
        records = collector.snapshot()
        allowed = authorized_passages(records, collection, policy)
        _, cited = format_citations(answer, allowed)
        isolated = set(re.findall(r"loadcase-\d{3}", answer)) == {request_id}
        if (not isolated or not cited or citation_numbers(answer) - cited or len(allowed) != len(records)
                or trace.snapshot()["grounding"] != "completed"):
            raise ValueError("Synthetic concurrency invariants did not hold")
        trace.finish("completed", allowed, cited)
        row.update(status="completed", context_isolated=isolated, cited_ids=sorted(cited))
    except Exception as error:
        row["error"] = error_details(error)
        trace.finish("failed")
    ended = time.perf_counter()
    row.update(service_seconds=ended - started, total_seconds=ended - submitted, execution=trace.snapshot())
    return row


def run_load(args, report, save):
    setup = time.perf_counter()
    import agent  # noqa: F401 - compile/import the graph before timing the burst
    import evidence  # noqa: F401 - avoid first-worker import costs in queue measurements
    from google import genai
    import httpx
    from access_control import PublicCollection
    from eval.evaluate_showcase import provenance
    from hybrid_search import get_reranker, search_chroma_hybrid
    from gemini_helpers import search_chroma
    from llm_router import GeminiRouter, RouterPolicy
    from showcase_corpus import prepare_showcase

    collection, _ = prepare_showcase(args.root)
    scoped = PublicCollection(collection)
    if args.search == "reranked":
        get_reranker()  # Explicit single-threaded cold load, outside measured burst.
    warmup = (search_chroma("solar wind battery unit costs", scoped) if args.search == "vector" else
              search_chroma_hybrid("solar wind battery unit costs", scoped, use_reranker=args.search == "reranked"))
    if not warmup:
        raise ValueError("Showcase warmup returned no public evidence")
    report.update(provenance=provenance(None, scoped), setup_seconds=time.perf_counter() - setup,
                  workload={"requests": args.requests, "concurrency": args.concurrency,
                            "provider_latency_ms": args.latency_ms, "fail_first_on_even_cases": args.fail_first,
                            "search": args.search}, rows=[])
    save()
    synthetic = SyntheticTransport(args.latency_ms / 1000, args.fail_first)
    original = genai.Client
    policy = RouterPolicy.from_environment()
    with httpx.Client(transport=httpx.MockTransport(synthetic.handle)) as http:
        def client(**kwargs):
            kwargs["http_options"] = kwargs["http_options"].model_copy(update={"httpx_client": http})
            return original(**kwargs)

        with patch("llm_router.genai.Client", side_effect=client):
            router = GeminiRouter(["synthetic-key-a", "synthetic-key-b"],
                                  models=["models/synthetic-primary", "models/synthetic-fallback"], policy=policy)
        try:
            started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futures = [pool.submit(load_request, f"loadcase-{index:03}", time.perf_counter(),
                                       router, collection, args.search) for index in range(args.requests)]
                for future in as_completed(futures):
                    report["rows"].append(future.result())
                    save()
            elapsed = time.perf_counter() - started
            actual = Counter(event["request_id"] for event in synthetic.events)
            for row in report["rows"]:
                expected = 3 + (args.fail_first if int(row["request_id"][-3:]) % 2 == 0 else 0)
                row["dispatch_accounting_matches"] = (actual[row["request_id"]] == expected
                    and row["execution"]["provider_attempts"] == actual[row["request_id"]])
                row["fallback_routing_matches"] = (
                    ("models/synthetic-fallback" in row["execution"]["provider_models"])
                    == (args.fail_first == 2 and int(row["request_id"][-3:]) % 2 == 0))
                if row["status"] == "completed" and not row["dispatch_accounting_matches"]:
                    row["status"] = "accounting_mismatch"
                if row["status"] == "completed" and not row["fallback_routing_matches"]:
                    row["status"] = "routing_mismatch"
            report.update(summary=summarize_load(report["rows"], elapsed),
                          synthetic_dispatches=len(synthetic.events),
                          synthetic_429s=sum(event["status"] == 429 for event in synthetic.events),
                          peak_synthetic_dispatch_concurrency=synthetic.peak_active)
        finally:
            for item in router.clients:
                item.close()
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["process_lifetime_peak_rss_bytes"] = rss * (1024 if platform.system() == "Linux" else 1)
    except ImportError:
        report["process_lifetime_peak_rss_bytes"] = None
    report["rows"].sort(key=lambda row: row["request_id"])
    return len(report["rows"]) == args.requests and all(row["status"] == "completed" for row in report["rows"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="kind", required=True)
    models = sub.add_parser("models", help="Check configured model metadata; never generate")
    models.add_argument("--live", action="store_true", required=True)
    models.add_argument("--output", type=Path, default=Path("eval/results/provider_models.json"))
    load = sub.add_parser("load", help="Local real-agent/SDK/retrieval test with synthetic provider transport")
    load.add_argument("--root", default="./showcase_data")
    load.add_argument("--requests", type=int, default=8)
    load.add_argument("--concurrency", type=int, default=2)
    load.add_argument("--latency-ms", type=int, default=50)
    load.add_argument("--fail-first", type=int, choices=(0, 1, 2), default=1)
    load.add_argument("--search", choices=("vector", "hybrid", "reranked"), default="reranked")
    load.add_argument("--output", type=Path, default=Path("eval/results/provider_concurrency.json"))
    args = parser.parse_args()
    if args.kind == "load" and not (1 <= args.requests <= 32 and 1 <= args.concurrency <= 4 and 0 <= args.latency_ms <= 2000):
        parser.error("Use 1–32 requests, 1–4 workers, and 0–2000 ms synthetic latency")
    for key in ("LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING", "ANONYMIZED_TELEMETRY", "GRADIO_ANALYTICS_ENABLED"):
        os.environ[key] = "false"
    os.environ["APP_MODE"] = "public_demo"
    os.environ["SUPABASE_ENABLED"] = "false"
    if args.kind == "load":
        os.environ.update(PYTHON_DOTENV_DISABLED="1", GEMINI_API_KEY="synthetic-load-placeholder",
                          GEMINI_API_KEY_2="", GOOGLE_GENAI_USE_VERTEXAI="false")
    from showcase_corpus import atomic_write

    report = {"kind": args.kind, "created_at": datetime.now(timezone.utc).isoformat(), "run_status": "running",
              "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "limits": ["Model metadata is not generation, quota, or tool-calling validation.",
                         "Synthetic-provider agent load is not real Gemini throughput or a Gradio/HTTP load test.",
                         "Warmup is outside burst timings; memory is process-lifetime peak, not per-request allocation."]}

    def save():
        atomic_write(args.output, (json.dumps(report, indent=2) + "\n").encode())

    save()
    try:
        if args.kind == "models":
            from config import settings
            from llm_router import GeminiRouter

            router = GeminiRouter(settings.GEMINI_API_KEYS, policy=settings.PROVIDER_POLICY)
            report.update(rows=[], model_roles=configured_models(router))

            def save_row(row):
                report["rows"].append(row)
                save()
                print(json.dumps(row))

            try:
                probe_models(router.clients, report["model_roles"],
                             min(10000, int(router.policy.call_timeout_seconds * 1000)), save_row)
            finally:
                for item in router.clients:
                    item.close()
            passed = all(row["status"] == "metadata_available" for row in report["rows"])
        else:
            passed = run_load(args, report, save)
        report["run_status"] = "completed" if passed else "failed"
    except Exception as error:
        report.update(run_status="blocked", setup_error=error_details(error))
    save()
    print(json.dumps({key: report.get(key) for key in (
        "kind", "run_status", "summary", "synthetic_dispatches", "synthetic_429s",
        "peak_synthetic_dispatch_concurrency", "process_lifetime_peak_rss_bytes", "setup_error")}, indent=2))
    return 0 if report["run_status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
