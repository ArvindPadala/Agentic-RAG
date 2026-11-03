"""Actual graph events and request outcomes with synthetic providers and data."""

import json
import threading
from unittest.mock import MagicMock, patch

import gradio as gr
import pytest
from fastapi.testclient import TestClient
from google.genai import types

from agent import build_agent_config, build_search_tool, run_agent_turn
from app import build_ui, make_chat_fn
from evidence import EvidenceCollector
from evidence import EvidenceHistory
from execution import ExecutionTrace, IncompleteResponseError, execution_status, render_execution
from live_guardrail import check_faithfulness
from query_optimizer import decompose_query
from tests.test_runtime_privacy import collection, ready_collection  # noqa: F401 - real fixture
from tests.test_supabase_memory import A, browser_request, sessions, snapshot


def reply(text, finish_reason="STOP"):
    return types.GenerateContentResponse(candidates=[types.Candidate(
        finish_reason=finish_reason,
        content=types.Content(role="model", parts=[types.Part(text=text)]))])


@pytest.mark.parametrize("payload,status", [
    ({"is_faithful": True, "reason": "Supported by the supplied passage."}, "completed"),
    ({"is_faithful": False, "reason": "The claim is not in the supplied passage."}, "flagged"),
    ({}, "unavailable"),
    ({"is_faithful": "false", "reason": "String verdict"}, "unavailable"),
    ({"is_faithful": 1, "reason": "Integer verdict"}, "unavailable"),
    ({"is_faithful": True}, "unavailable"),
    ({"is_faithful": True, "reason": " "}, "unavailable"),
    ([], "unavailable"),
])
def test_grounding_verdicts_are_strict_and_missing_fields_do_not_pass(payload, status):
    client = MagicMock()
    client.models.generate_content.return_value = reply(json.dumps(payload))
    assert check_faithfulness("Answer", ["Passage"], client).status == status


def test_grounding_api_and_json_errors_are_unavailable_without_exposing_error_details():
    client = MagicMock()
    client.models.generate_content.side_effect = RuntimeError("private-provider-error-sentinel")
    verdict = check_faithfulness("Answer", [], client)
    assert verdict.status == "unavailable"
    assert "private-provider" not in verdict.reason
    client.models.generate_content.side_effect = None
    client.models.generate_content.return_value = reply("not valid JSON")
    assert check_faithfulness("Answer", [], client).status == "unavailable"


@pytest.mark.parametrize("queries", [[], [""], ["a", "b", "c", "d"], [3], ["x" * 513]])
def test_decomposition_fallback_is_recorded_and_original_query_preserved(queries):
    client, trace = MagicMock(), ExecutionTrace()
    client.models.generate_content.return_value = reply(json.dumps(queries))
    assert decompose_query("Original question", client, execution=trace) == ["Original question"]
    assert trace.snapshot()["decomposition"] == "fallback"
    assert trace.snapshot()["subqueries"] == []


def test_actual_graph_records_stages_queries_searches_and_current_turn_grounding(collection):
    stages, trace, collector = [], ExecutionTrace("Standard Vector Search"), EvidenceCollector()
    trace.on_stage = stages.append
    search, tool = build_search_tool(collection, None, None, None, evidence_collector=collector, execution=trace)
    tool_call = types.GenerateContentResponse(candidates=[types.Candidate(finish_reason="STOP", content=types.Content(
        role="model", parts=[types.Part(function_call=types.FunctionCall(
            name="search_knowledge_base", args={"query": "showcase"}))]))])
    client = MagicMock()
    client.models.generate_content.side_effect = [
        reply('["first focused query", "second focused query"]'), tool_call,
        reply("Answer [1]."), reply('{"is_faithful":true,"reason":"Supported."}'),
    ]
    old = types.Content(role="user", parts=[types.Part.from_function_response(
        name="search_knowledge_base", response={"result": "historical-context-sentinel"})])
    answer = run_agent_turn("Question", [old], client, build_agent_config(tool, {}, structured_citations=True),
                            {"search_knowledge_base": search}, use_decomposition=True,
                            use_guardrail=True, execution=trace)
    summary = trace.snapshot()
    assert summary["subqueries"] == ["first focused query", "second focused query"]
    assert summary["searches"] == 1 and summary["generation_calls"] == 2
    assert summary["grounding"] == "completed"
    assert "Optimising search queries" in stages
    assert "Searching the authorized reports" in stages
    assert "Checking the answer against retrieved evidence" in stages
    assert "Completed" in answer and "not a guarantee" in answer
    judge_input = client.models.generate_content.call_args.kwargs["contents"][0]
    assert "historical-context-sentinel" not in judge_input
    assert "Public showcase passage" in judge_input


def test_disabled_check_has_no_judge_call_and_empty_responses_fail_cleanly():
    client, trace = MagicMock(), ExecutionTrace()
    client.models.generate_content.return_value = reply("Boundary response")
    result = run_agent_turn("Question", [], client, build_agent_config(types.Tool(), {}), {}, execution=trace)
    assert result == "Boundary response"
    assert trace.snapshot()["grounding"] == "not_run"
    assert client.models.generate_content.call_count == 1
    client.models.generate_content.return_value = types.GenerateContentResponse(candidates=[])
    with pytest.raises(RuntimeError, match="no usable response"):
        run_agent_turn("Question", [], client, build_agent_config(types.Tool(), {}), {})


@pytest.mark.parametrize("reason,recorded", [
    ("MAX_TOKENS", "MAX_TOKENS"), ("SAFETY", "SAFETY"), ("OTHER", "OTHER"), (None, "UNKNOWN"),
])
def test_non_normal_finish_withholds_text_skips_grounding_and_restores_history(reason, recorded):
    client, trace = MagicMock(), ExecutionTrace()
    client.models.generate_content.return_value = reply("partial-private-response-sentinel", reason)
    previous = types.Content(role="model", parts=[types.Part(text="Previous completed answer")])
    history = [previous]
    with pytest.raises(IncompleteResponseError) as raised:
        run_agent_turn("Question", history, client, build_agent_config(types.Tool(), {}), {},
                       use_guardrail=True, execution=trace)
    assert raised.value.finish_reason == recorded
    assert "partial-private" not in str(raised.value)
    assert history == [previous]
    assert trace.snapshot()["finish_reason"] == recorded
    assert trace.snapshot()["status"] == "incomplete"
    assert trace.snapshot()["grounding"] == "not_run"
    assert client.models.generate_content.call_count == 1


def test_partial_tool_call_never_executes():
    client, trace, tool = MagicMock(), ExecutionTrace(), MagicMock()
    client.models.generate_content.return_value = types.GenerateContentResponse(candidates=[types.Candidate(
        finish_reason="MAX_TOKENS", content=types.Content(role="model", parts=[
            types.Part.from_function_call(name="search_knowledge_base", args={"query": "partial"})]))])
    with pytest.raises(IncompleteResponseError):
        run_agent_turn("Question", [], client, build_agent_config(types.Tool(), {}),
                       {"search_knowledge_base": tool}, use_guardrail=True, execution=trace)
    tool.assert_not_called()
    assert client.models.generate_content.call_count == 1


def test_incomplete_final_answer_rolls_back_earlier_successful_tool_cycle():
    client, trace, tool = MagicMock(), ExecutionTrace(), MagicMock(return_value="Public evidence")
    client.models.generate_content.side_effect = [
        types.GenerateContentResponse(candidates=[types.Candidate(finish_reason="STOP", content=types.Content(
            role="model", parts=[types.Part.from_function_call(name="search_knowledge_base", args={"query": "public"})]))]),
        reply("unfinished final response", "MAX_TOKENS"),
    ]
    previous = types.Content(role="model", parts=[types.Part(text="Previous completed answer")])
    history = [previous]
    with pytest.raises(IncompleteResponseError):
        run_agent_turn("Question", history, client, build_agent_config(types.Tool(), {}),
                       {"search_knowledge_base": tool}, use_guardrail=True, execution=trace)
    tool.assert_called_once_with(query="public")
    assert history == [previous]
    assert trace.snapshot()["finish_reason"] == "MAX_TOKENS"
    assert client.models.generate_content.call_count == 2


@pytest.mark.parametrize("error", [RuntimeError("provider failure"), KeyboardInterrupt()])
def test_provider_failure_or_operator_interrupt_rolls_back_before_cli_can_save(error):
    client, trace, tool = MagicMock(), ExecutionTrace(), MagicMock(return_value="Public evidence")
    client.models.generate_content.side_effect = [
        types.GenerateContentResponse(candidates=[types.Candidate(finish_reason="STOP", content=types.Content(
            role="model", parts=[types.Part.from_function_call(name="search_knowledge_base", args={"query": "public"})]))]),
        error,
    ]
    previous = types.Content(role="model", parts=[types.Part(text="Previous completed context")])
    history = [previous]
    with pytest.raises(type(error)):
        run_agent_turn("Question", history, client, build_agent_config(types.Tool(), {}),
                       {"search_knowledge_base": tool}, execution=trace)
    assert history == [previous]
    assert trace.snapshot()["status"] == "failed"
    assert trace.snapshot()["finish_reason"] == "UNKNOWN"


@pytest.mark.parametrize("reason", ["MAX_TOKENS", None])
def test_incomplete_structured_json_cannot_pass_grounding_or_decomposition(reason):
    client, trace = MagicMock(), ExecutionTrace()
    client.models.generate_content.return_value = reply('{"is_faithful":true,"reason":"Looks complete."}', reason)
    assert check_faithfulness("Answer", ["Passage"], client).status == "unavailable"
    client.models.generate_content.return_value = reply('["apparently valid query"]', reason)
    assert decompose_query("Original question", client, execution=trace) == ["Original question"]
    assert trace.snapshot()["decomposition"] == "fallback"


@pytest.mark.parametrize("parts", [
    [types.Part(text=" ")], [types.Part(text="hidden thought sentinel", thought=True)],
])
def test_normal_stop_without_visible_answer_fails_and_preserves_history(parts):
    client, trace, history = MagicMock(), ExecutionTrace(), []
    client.models.generate_content.return_value = types.GenerateContentResponse(candidates=[types.Candidate(
        finish_reason="STOP", content=types.Content(role="model", parts=parts))])
    with pytest.raises(RuntimeError, match="no usable response"):
        run_agent_turn("Question", history, client, build_agent_config(types.Tool(), {}), {},
                       use_guardrail=True, execution=trace)
    assert history == []
    assert trace.snapshot()["status"] == "failed"
    assert client.models.generate_content.call_count == 1


@pytest.mark.parametrize("return_evidence", [False, True])
def test_web_incomplete_response_preserves_completed_context_and_hides_partial_text(collection, return_evidence):
    client, trace = MagicMock(), ExecutionTrace()
    client.models.generate_content.return_value = reply("partial-private-response-sentinel", "MAX_TOKENS")
    previous = types.Content(role="model", parts=[types.Part(text="Previous completed context")])
    chat = make_chat_fn(client, {}, "unused.json", None, collection, None, return_evidence=return_evidence)
    output = chat("Question", [], [previous], "Standard Vector Search", False, True, execution=trace)
    assert output[1] == [previous]
    assert "output limit" in output[0][-1]["content"]
    assert "partial-private" not in str(output)
    if return_evidence:
        assert output[5] is None
    summary = trace.snapshot()
    assert summary["status"] == "incomplete" and summary["grounding"] == "not_run"
    assert "Answer ready" not in execution_status(summary)
    assert "MAX_TOKENS" in render_execution(summary)
    assert client.models.generate_content.call_count == 1


def test_execution_display_escapes_queries_and_does_not_infer_costs():
    trace = ExecutionTrace("Vector")
    trace.set_decomposition("completed", ["<script>bad()</script>"])
    trace.finish("completed")
    display = render_execution(trace.snapshot())
    assert "<script>" not in display
    assert "&lt;script&gt;" in display
    assert "No token/cost totals are inferred" in display


def test_failed_retrieval_is_counted_and_excluded_from_grounding_context():
    trace, collection = ExecutionTrace(), MagicMock()
    search, tool = build_search_tool(collection, None, None, None, execution=trace)
    tool_call = types.GenerateContentResponse(candidates=[types.Candidate(finish_reason="STOP", content=types.Content(
        role="model", parts=[types.Part(function_call=types.FunctionCall(
            name="search_knowledge_base", args={"query": "query"}))]))])
    client = MagicMock()
    client.models.generate_content.side_effect = [tool_call, reply("No evidence is available."),
        reply('{"is_faithful":true,"reason":"The answer acknowledges missing evidence."}')]
    with patch("agent.search_chroma", side_effect=RuntimeError("private-search-error")):
        result = run_agent_turn("Question", [], client, build_agent_config(tool, {}),
                                {"search_knowledge_base": search}, execution=trace, use_guardrail=True)
    assert trace.snapshot()["search_errors"] == 1
    assert trace.snapshot()["searches"] == 1
    assert "private-search-error" not in result
    judge = client.models.generate_content.call_args.kwargs["contents"][0]
    assert "NO CONTEXT RETRIEVED" in judge
    assert "Error: Search" not in judge


def test_streamed_private_details_cannot_return_after_lifecycle_change():
    backend = MagicMock()
    backend.load_memory.return_value = snapshot()
    service = sessions(backend)
    demo = build_ui(None, {}, "unused.json", None, ready_collection(), None, private_sessions=service)
    stream = next(fn.fn for fn in demo.fns.values() if fn.fn and fn.fn.__name__ == "stream_submit")
    scope = {"generation": None, "evidence": EvidenceHistory(), "execution": None}
    entered, release = threading.Event(), threading.Event()

    def delayed(**kwargs):
        kwargs["execution"].set_decomposition("completed", ["private-subquery-sentinel"])
        kwargs["execution"].stage("Generating a response or next search")
        entered.set()
        assert release.wait(10)
        return "private-answer-sentinel"

    try:
        with patch("app.run_agent_turn", side_effect=delayed):
            iterator = stream("Question", [], [], "Standard Vector Search", True, False,
                              A, scope, browser_request(service))
            first = next(iterator)
            assert first[11] == "Preparing authorized research"
            assert entered.wait(10)
            with scope["evidence"].lock:
                scope["generation"] = "ended-session"
                scope["evidence"].clear()
            release.set()
            remaining = list(iterator)
            assert "private-subquery-sentinel" not in str(remaining)
            assert "private-answer-sentinel" not in str(remaining)
            assert scope["execution"] is None
    finally:
        release.set()
        demo.close()


def test_ui_failure_status_and_clear_details_preserve_previous_context():
    demo = build_ui(None, {}, "unused.json", None, ready_collection(), None)
    app = gr.routes.App.create_app(demo)

    def fail(**kwargs):
        kwargs["conversation_history"].append("unfinished turn")
        raise RuntimeError("429 provider-private-error-sentinel")

    try:
        with TestClient(app) as client, patch("app.run_agent_turn", side_effect=fail):
            result = client.post("/gradio_api/api/submit", json={
                "data": ["Question", [], None, None, None, None, None, None], "session_hash": "status-test"})
            assert result.status_code == 200
            data = result.json()["data"]
            assert "rate/quota" in data[11]
            assert "daily" in data[11] and "temporary" in data[11]
            assert "provider-private-error" not in result.text
            cleared = client.post("/gradio_api/api/clear_chat", json={
                "data": [None, None], "session_hash": "status-test"})
            assert "Run a research question" in cleared.json()["data"][12]
    finally:
        demo.close()
