"""Evaluate only the checksum-pinned public IPCC showcase, never the legacy index.

Run from the repository root: python -m eval.evaluate_showcase --help
Offline retrieval is the default; --live explicitly enables Gemini generation.
"""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import textwrap
import time
import unicodedata


DATASET = Path(__file__).with_name("ipcc_benchmark.json")
MODES = ("vector", "hybrid", "reranked")
GROUPED_REFERENCE_RE = re.compile(r"\[(?:\d+\s*,\s*)+\d+\]")


def normalize(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    text = text.translate(str.maketrans({"–": "-", "−": "-", "‑": "-", "\u00ad": ""}))
    return " ".join(text.split())


def load_benchmark(path, sources, parser_version):
    raw = path.read_bytes()
    data = json.loads(raw)
    pins = {source["id"]: source["sha256"] for source in sources}
    if data.get("schema_version") != 1 or data.get("source_sha256") != pins:
        raise ValueError("Benchmark source pins/schema do not match the current showcase")
    if data.get("parser_version") != parser_version:
        raise ValueError("Benchmark parser version does not match the current showcase")
    cases = data.get("cases", [])
    if not cases or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Benchmark requires nonempty, unique case IDs")
    for case in cases:
        if case["kind"] not in {"supported", "cross_report", "unsupported"}:
            raise ValueError("Unknown benchmark case kind")
        if not case["question"].strip() or not case["review_guidance"].strip():
            raise ValueError("Question and answer-review guidance are required")
        groups = case["evidence"]
        if bool(groups) == (case["kind"] == "unsupported"):
            raise ValueError("Supported cases need evidence; unsupported cases must not have it")
        for group in groups:
            if (group["source_id"] not in pins or len(normalize(group["quote"])) < 20
                    or type(group.get("pdf_page")) is not int or group["pdf_page"] < 1):
                raise ValueError("Evidence requires a pinned source and a specific quote")
        if case["kind"] == "cross_report" and len({g["source_id"] for g in groups}) < 2:
            raise ValueError("Cross-report cases require evidence from distinct reports")
    return data, hashlib.sha256(raw).hexdigest()


def resolve_evidence(cases, chunks):
    """Resolve source-qualified literal witnesses before retrieval, never from hits."""
    gold = {}
    for case in cases:
        groups = []
        for group in case["evidence"]:
            matches = {chunk["chunk_id"] for chunk in chunks
                       if chunk["chunk_id"].startswith(group["source_id"] + ":")
                       and chunk["page"] + 1 == group["pdf_page"]
                       and normalize(group["quote"]) in normalize(chunk["text"])}
            if not matches:
                raise ValueError(f"Unresolved evidence for {case['id']}: {group['source_id']}")
            groups.append(matches)
        gold[case["id"]] = groups
    return gold


def score_evidence(results, groups):
    if not groups:
        return None  # Negative questions do not have meaningful retrieval recall.
    ids = [result["chunk_id"] for result in results]
    hit_groups = sum(bool(set(ids) & group) for group in groups)
    ranks = [rank for rank, chunk_id in enumerate(ids, 1) if any(chunk_id in group for group in groups)]
    return {"annotated_group_coverage": hit_groups / len(groups),
            "any_annotated_hit": bool(hit_groups), "all_annotated_groups_hit": hit_groups == len(groups),
            "reciprocal_rank_first_annotated_hit": 1 / min(ranks) if ranks else 0.0}


def summarize(rows):
    summary = {}
    for mode in MODES:
        selected = [row for row in rows if row["mode"] == mode and row.get("score") is not None]
        successful = [row for row in selected if row["status"] == "completed"]
        scores = [row["score"] for row in selected]
        summary[mode] = {
            "scored_cases": len(selected), "completed_cases": len(successful),
            "failed_cases": len(selected) - len(successful),
            "mean_annotated_group_coverage_at_5": mean(s["annotated_group_coverage"] for s in scores),
            "any_annotated_hit_rate_at_5": mean(s["any_annotated_hit"] for s in scores),
            "all_annotated_groups_hit_rate_at_5": mean(s["all_annotated_groups_hit"] for s in scores),
            "mrr_first_annotated_hit_at_5": mean(s["reciprocal_rank_first_annotated_hit"] for s in scores),
            "preview_successes": sum(row.get("preview_rendered", False) for row in successful),
            "search_seconds_total": sum(row["search_seconds"] for row in selected),
        }
    return summary


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def error_details(error):
    # Provider exceptions may contain request details. Do not serialize str(error).
    code = getattr(error, "code", None)
    return {"type": type(error).__name__, "code": code if isinstance(code, int) else None}


def retrieve(case, mode, collection, corpus, groups):
    from gemini_helpers import search_chroma
    from hybrid_search import search_chroma_hybrid

    started = time.perf_counter()
    row = {"case_id": case["id"], "mode": mode, "status": "failed",
           "score": score_evidence([], groups), "results": []}
    try:
        results = (search_chroma(case["question"], collection, n_results=5) if mode == "vector" else
                   search_chroma_hybrid(case["question"], collection, n_results=5,
                                        use_reranker=mode == "reranked"))
        row.update(status="completed", score=score_evidence(results, groups), results=results)
    except Exception as error:
        row["error"] = error_details(error)
    row["search_seconds"] = time.perf_counter() - started
    if row["status"] == "completed":
        try:
            row["preview_rendered"] = bool(row["results"] and corpus.render_citation(row["results"][0]))
        except Exception as error:
            row["preview_rendered"] = False
            row["preview_error"] = error_details(error)
    return row


def live_case(case, collection, groups, model, mode, guardrail):
    from agent import build_agent_config, build_search_tool, run_agent_turn
    from config import settings
    from access_control import RuntimePolicy
    from evidence import CITATION_RE, EvidenceCollector, authorized_passages, format_citations
    from execution import ExecutionTrace
    from llm_router import GeminiRouter

    # Fresh router/history/evidence per case. No user memory, Supabase or S3.
    router = GeminiRouter(settings.GEMINI_API_KEYS, models=[model])
    collector = EvidenceCollector()
    execution = ExecutionTrace(mode)
    policy = RuntimePolicy("public_demo")
    search, tool = build_search_tool(collection, router, None, None,
                                    use_hybrid=mode != "vector", use_reranker=mode == "reranked",
                                    policy=policy, evidence_collector=collector, execution=execution)
    row = {"case_id": case["id"], "status": "failed", "human_review": "pending"}
    try:
        answer = run_agent_turn(case["question"], [], router,
                                build_agent_config(tool, {}, structured_citations=True),
                                {"search_knowledge_base": search}, model=model,
                                use_decomposition=case["kind"] == "cross_report",
                                use_guardrail=guardrail, execution=execution)
        records = collector.snapshot()
        allowed = authorized_passages(records, collection, policy)
        if len(allowed) != len(records):
            raise ValueError("Evidence authorization changed")
        _, cited = format_citations(answer, allowed)
        refs = {int(match.group(1) or match.group(2)) for match in CITATION_RE.finditer(answer)}
        captured = [{"chunk_id": p.chunk_id} for p in allowed]
        cited_chunks = [{"chunk_id": p.chunk_id} for p in allowed if p.number in cited]
        row.update(status="completed", answer=answer, cited_ids=sorted(cited),
                   invalid_citation_ids=sorted(refs - set(cited)),
                   unparsed_grouped_citations=GROUPED_REFERENCE_RE.findall(answer),
                   citation_id_precision=len(cited) / len(refs) if refs else None,
                   has_citations=bool(cited), captured_evidence_score=score_evidence(captured, groups),
                   cited_evidence_score=score_evidence(cited_chunks, groups),
                   evidence=[{"number": p.number, "chunk_id": p.chunk_id, "source_key": p.source_key,
                              "title": p.title, "pdf_page": p.page + 1, "excerpt": p.excerpt}
                             for p in allowed])
        execution.finish("completed", records, cited)
    except Exception as error:
        row["error"] = error_details(error)
        execution.finish("failed")
    row["execution"] = execution.snapshot()
    return row


def provenance(dataset_hash, collection):
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                         cwd=DATASET.parent.parent).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    files = ["eval/evaluate_showcase.py", "showcase_corpus.py", "manage_public_corpus.py",
             "access_control.py", "gemini_helpers.py", "hybrid_search.py", "agent.py", "config.py",
             "evidence.py", "execution.py", "query_optimizer.py", "live_guardrail.py", "llm_router.py"]
    return {"dataset_sha256": dataset_hash, "git_commit": commit,
            "implementation_sha256": {name: hashlib.sha256((DATASET.parent.parent / name).read_bytes()).hexdigest()
                                      for name in files}, "python": platform.python_version(),
            "platform": platform.system(), "machine": platform.machine(),
            "packages": {name: version(name) for name in ("chromadb", "sentence-transformers", "PyMuPDF")},
            "collection": collection.name, "chunk_count": collection.count(),
            "embedding": "Chroma default all-MiniLM-L6-v2", "reranker": "ms-marco-MiniLM-L-6-v2 (CPU)"}


def review_report(path):
    """Render saved public-benchmark answers for review without rerunning inference."""
    report = json.loads(path.read_text())
    print(json.dumps({"run_status": report.get("run_status", "unknown"), "retrieval_summary": report["retrieval_summary"],
                      "live_cases_saved": len(report["live"])}, indent=2))
    for row in report["live"]:
        execution = row.get("execution", {})
        print(f"\n## {row['case_id']} — {row['status']}")
        print(json.dumps({key: row.get(key) for key in (
            "citation_id_precision", "invalid_citation_ids", "captured_evidence_score", "cited_evidence_score")}, indent=2))
        print("Unparsed grouped references: " + json.dumps(GROUPED_REFERENCE_RE.findall(row.get("answer", ""))))
        print(json.dumps({key: execution.get(key) for key in (
            "elapsed", "searches", "generation_calls", "decomposition", "grounding", "grounding_reason")}, indent=2))
        for paragraph in row.get("answer", "").splitlines():
            print(textwrap.fill(paragraph, width=100))
        print("Cited source pages:")
        for passage in row.get("evidence", []):
            if passage["number"] in row.get("cited_ids", []):
                print(f"[{passage['number']}] {passage['title']} — PDF page {passage['pdf_page']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="./showcase_data")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output", type=Path, default=Path("eval/results/ipcc_evaluation.json"))
    parser.add_argument("--find", action="append", default=[], help="Inspect pinned public passages; no indexing/generation")
    parser.add_argument("--review-report", type=Path, help="Read a saved benchmark report without rerunning evaluation")
    parser.add_argument("--validate-only", action="store_true", help="Check source pins and quote witnesses without indexing")
    parser.add_argument("--live", action="store_true", help="Also call Gemini on selected cases; may consume quota")
    parser.add_argument("--case", action="append", default=[], help="Select case IDs; repeat for multiple cases")
    parser.add_argument("--model", default="models/gemini-3.6-flash")
    parser.add_argument("--live-mode", choices=MODES, default="reranked")
    parser.add_argument("--guardrail", action="store_true", help="Enable the actual LLM grounding check for live cases")
    args = parser.parse_args()
    if args.review_report:
        if args.live or args.find or args.validate_only:
            parser.error("--review-report cannot be combined with execution modes")
        review_report(args.review_report)
        return 0
    if args.live and (args.find or args.validate_only):
        parser.error("--live cannot be combined with inspection/validation-only modes")
    if args.guardrail and not args.live:
        parser.error("--guardrail requires --live")
    # Disable content-bearing remote traces before importing runtime modules.
    for key in ("LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING", "ANONYMIZED_TELEMETRY", "GRADIO_ANALYTICS_ENABLED"):
        os.environ[key] = "false"
    os.environ["APP_MODE"] = "public_demo"
    os.environ["SUPABASE_ENABLED"] = "false"
    if not args.live:
        os.environ["PYTHON_DOTENV_DISABLED"] = "1"

    from showcase_corpus import (PARSER_VERSION, atomic_write, curated_sources,
                                 download_pdf, extract_chunks, prepare_showcase)
    from access_control import PublicCollection

    root = Path(args.root).resolve()
    sources = curated_sources()
    chunks = []
    for source in sources:
        path, digest = download_pdf(source, root)
        chunks.extend(extract_chunks(path, source, digest))
    if args.find:
        for term in args.find:
            matches = [c for c in chunks if normalize(term) in normalize(c["text"])]
            print(json.dumps({"term": term, "matches": len(matches), "examples": [
                {"chunk_id": c["chunk_id"], "pdf_page": c["page"] + 1, "text": c["text"]}
                for c in matches[:4]]}, ensure_ascii=False, indent=2))
        return 0
    dataset, dataset_hash = load_benchmark(args.dataset, sources, PARSER_VERSION)
    all_cases = dataset["cases"]
    if set(args.case) - {case["id"] for case in all_cases}:
        parser.error("Unknown --case ID")
    cases = [case for case in all_cases if not args.case or case["id"] in args.case]
    gold = resolve_evidence(cases, chunks)
    by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    witnesses = {case["id"]: [[{"chunk_id": cid, "pdf_page": by_id[cid]["page"] + 1,
                                "text": by_id[cid]["text"]} for cid in sorted(group)]
                              for group in gold[case["id"]]] for case in cases}
    if args.validate_only:
        print(json.dumps({"validated_cases": len(cases), "witnesses": witnesses}, ensure_ascii=False, indent=2))
        return 0
    collection, corpus = prepare_showcase(root)
    scoped = PublicCollection(collection)
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "run_status": "running",
              "provenance": provenance(dataset_hash, scoped), "source_sha256": dataset["source_sha256"],
              "parser_version": PARSER_VERSION, "cases": cases, "witnesses": witnesses,
              "retrieval": [], "live": [], "live_requested": args.live,
              "live_model": args.model if args.live else None, "live_mode": args.live_mode if args.live else None,
              "guardrail_requested": args.guardrail if args.live else False,
              "limitations": ["Small development benchmark; not a held-out or comprehensive accuracy estimate.",
                              "Annotated witnesses are not exhaustive relevance labels; coverage is not full recall.",
                              "Citation ID validity is provenance, not semantic entailment.",
                              "Unsupported-answer correctness and factual claims require human review.",
                              "Timings include cold starts; no load/capacity or GPU comparison is implied."]}

    def save():
        report["retrieval_summary"] = summarize(report["retrieval"])
        atomic_write(args.output, (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode())

    save()
    for mode in MODES:
        for case in cases:
            if not gold[case["id"]]:
                continue
            row = retrieve(case, mode, scoped, corpus, gold[case["id"]])
            report["retrieval"].append(row)
            save()
            print(f"{mode} {case['id']}: {row['status']} {row['score']}")
    if args.live:
        for case in cases:
            try:
                row = live_case(case, collection, gold[case["id"]], args.model, args.live_mode, args.guardrail)
            except Exception as error:
                row = {"case_id": case["id"], "status": "failed", "error": error_details(error),
                       "human_review": "pending"}
            report["live"].append(row)
            save()
            print(f"live {case['id']}: {row['status']}")
            if row["status"] != "completed":
                break  # Do not burn quota repeating a provider/configuration failure.
    failed = any(row["status"] != "completed" or not row.get("preview_rendered", False)
                 for row in report["retrieval"])
    failed = failed or (args.live and (len(report["live"]) != len(cases)
                                      or any(row["status"] != "completed" for row in report["live"])))
    report["run_status"] = "failed" if failed else "completed"
    save()
    print(json.dumps(report["retrieval_summary"], indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
