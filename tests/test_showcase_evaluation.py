"""Benchmark isolation, scoring denominators, and real agent/evidence integration."""

from copy import deepcopy
import json
from unittest.mock import MagicMock, patch

from google.genai import types
import pytest

from eval.evaluate_showcase import (DATASET, error_details, live_case, load_benchmark,
                                    resolve_evidence, score_evidence, summarize)
from showcase_corpus import PARSER_VERSION, curated_sources


def benchmark():
    return load_benchmark(DATASET, curated_sources(), PARSER_VERSION)[0]


def test_benchmark_rejects_changed_sources_parser_and_invalid_negative_labels(tmp_path):
    data = benchmark()
    sources = deepcopy(curated_sources())
    sources[0]["sha256"] = "changed"
    with pytest.raises(ValueError, match="pins"):
        load_benchmark(DATASET, sources, PARSER_VERSION)
    with pytest.raises(ValueError, match="parser"):
        load_benchmark(DATASET, curated_sources(), "changed")
    data["cases"][-1]["evidence"] = data["cases"][0]["evidence"]
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="unsupported"):
        load_benchmark(invalid, curated_sources(), PARSER_VERSION)


def test_reference_resolution_requires_both_pinned_source_and_exact_file_page():
    case = benchmark()["cases"][0]
    group = case["evidence"][0]
    wrong_source = {"chunk_id": "other-source:hash:chunk", "page": group["pdf_page"] - 1,
                    "text": group["quote"]}
    wrong_page = {**wrong_source, "chunk_id": group["source_id"] + ":hash:chunk", "page": 99}
    with pytest.raises(ValueError, match="Unresolved"):
        resolve_evidence([case], [wrong_source, wrong_page])
    right = {**wrong_page, "page": group["pdf_page"] - 1}
    assert resolve_evidence([case], [wrong_source, wrong_page, right]) == {
        case["id"]: [{right["chunk_id"]}]}


def test_cross_report_groups_are_scored_separately_without_duplicate_rank_inflation():
    groups = [{"report-a:passage"}, {"report-b:passage"}]
    rows = [{"chunk_id": cid} for cid in ["unrelated", "unrelated", "report-a:passage"]]
    score = score_evidence(rows, groups)
    assert score["annotated_group_coverage"] == 0.5
    assert score["all_annotated_groups_hit"] is False
    assert score["reciprocal_rank_first_annotated_hit"] == pytest.approx(1 / 3)
    assert score_evidence(rows, []) is None


def test_retrieval_failure_stays_in_aggregate_denominator_and_empty_modes_are_null():
    groups = [{"witness"}]
    rows = [{"mode": "vector", "status": "completed", "search_seconds": 0.1,
             "score": score_evidence([{"chunk_id": "witness"}], groups), "preview_rendered": True},
            {"mode": "vector", "status": "failed", "search_seconds": 0.2,
             "score": score_evidence([], groups)}]
    summary = summarize(rows)
    assert summary["vector"]["scored_cases"] == 2
    assert summary["vector"]["failed_cases"] == 1
    assert summary["vector"]["mean_annotated_group_coverage_at_5"] == 0.5
    assert summary["hybrid"]["mean_annotated_group_coverage_at_5"] is None


def test_live_evaluation_uses_actual_agent_and_only_counts_authorized_citation_ids():
    case = benchmark()["cases"][0]
    source_key = "showcase/hash/fixture.pdf"
    result = {"chunk_id": "fixture", "source_document": "Fixture report", "score": 1.0,
              "page": 0, "bbox": [0.1, 0.1, 0.5, 0.5], "chunk_type": "text",
              "text": "Human influence warmed the climate.", "source_pdf_key": source_key}
    collection = MagicMock()
    collection.get.return_value = {"ids": ["fixture"], "metadatas": [{"source_pdf_key": source_key, "page": 0}]}
    router = MagicMock()
    router.models.generate_content.side_effect = [
        types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(
            role="model", parts=[types.Part.from_function_call(name="search_knowledge_base", args={"query": "human"})]))]),
        types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(
            role="model", parts=[types.Part(text="Human influence warmed the climate [1]. Invented reference [999]. Group [1, 999].")]))]),
    ]
    with patch("llm_router.GeminiRouter", return_value=router), patch("agent.search_chroma", return_value=[result]):
        row = live_case(case, collection, [{"fixture"}], "fixture-model", "vector", False)
    assert row["status"] == "completed"
    assert row["cited_ids"] == [1]
    assert row["invalid_citation_ids"] == [999]
    assert row["unparsed_grouped_citations"] == ["[1, 999]"]
    assert row["citation_id_precision"] == 0.5
    assert row["cited_evidence_score"]["annotated_group_coverage"] == 1
    assert row["execution"]["searches"] == 1
    assert row["execution"]["generation_calls"] == 2
    assert row["human_review"] == "pending"


def test_live_failures_redact_provider_messages():
    error = RuntimeError("private request payload or credential must not be serialized")
    assert error_details(error) == {"type": "RuntimeError", "code": None}


def test_cli_saves_failure_and_stops_before_next_live_case(tmp_path, monkeypatch):
    from eval import evaluate_showcase

    output = tmp_path / "report.json"
    selected = [case for case in benchmark()["cases"] if case["kind"] == "unsupported"]
    monkeypatch.setattr("sys.argv", ["evaluate_showcase", "--live", "--output", str(output),
                                    "--case", selected[0]["id"], "--case", selected[1]["id"]])
    for key in ("APP_MODE", "SUPABASE_ENABLED", "LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING",
                "ANONYMIZED_TELEMETRY", "GRADIO_ANALYTICS_ENABLED"):
        monkeypatch.setenv(key, "false" if key != "APP_MODE" else "public_demo")
    failure = RuntimeError("secret provider request text")
    with patch("showcase_corpus.download_pdf", return_value=(None, "hash")), patch(
            "showcase_corpus.extract_chunks", return_value=[]), patch(
            "showcase_corpus.prepare_showcase", return_value=(MagicMock(), MagicMock())), patch(
            "eval.evaluate_showcase.provenance", return_value={}), patch(
            "eval.evaluate_showcase.live_case", side_effect=failure) as live:
        assert evaluate_showcase.main() == 1
    report = json.loads(output.read_text())
    assert report["run_status"] == "failed"
    assert len(report["live"]) == 1
    assert report["live"][0]["case_id"] == selected[0]["id"]
    assert live.call_count == 1
    assert "secret provider" not in output.read_text()
