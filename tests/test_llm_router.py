"""Deterministic provider budgets/routing, plus an actual SDK mock-transport check."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from email.utils import format_datetime
from types import SimpleNamespace
import threading
from unittest.mock import MagicMock, patch

from google import genai
from google.genai import types
from google.genai.errors import APIError
import httpx
import pytest

from agent import build_agent_config, run_agent_turn
from app import make_chat_fn
from execution import ExecutionTrace
from llm_router import (GeminiRouter, ProviderAttemptsExceeded, ProviderDeadlineExceeded,
                        ProviderRequestError, ProviderTimeoutError, RouterPolicy, provider_request)
from tests.test_runtime_privacy import ready_collection


class Clock:
    def __init__(self):
        self.now = 100.0
        self.sleeps = []

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    clock = Clock()
    monkeypatch.setattr("llm_router.time", SimpleNamespace(
        monotonic=lambda: clock.now, sleep=clock.sleep, time=lambda: 1700000000 + clock.now - 100))
    monkeypatch.setattr("llm_router.random", SimpleNamespace(uniform=lambda lower, upper: upper))
    return clock


@pytest.fixture
def clients():
    with patch("llm_router.genai.Client", side_effect=lambda **kwargs: MagicMock()) as factory:
        yield factory


def router(policy=None, *, keys=("key-a", "key-b"), models=("model-a", "model-b")):
    return GeminiRouter(list(keys), list(models), policy=policy or RouterPolicy())


def response(text="Complete answer"):
    return types.GenerateContentResponse(candidates=[types.Candidate(
        finish_reason="STOP", content=types.Content(role="model", parts=[types.Part(text=text)]))])


def failure(code, *, retry_after=None, details=None):
    headers = {"Retry-After": retry_after} if retry_after is not None else {}
    body = {"error": {"code": code, "message": "private-provider-message-sentinel", "details": details or []}}
    return APIError(code, body, httpx.Response(code, headers=headers, json=body))


def test_deduplicates_routes_and_disables_sdk_retries(clients):
    provider = router(keys=("key-a", "key-a", "key-b"), models=("model-a", "model-a"))
    assert len(provider.clients) == 2
    assert provider.fallback_models == ("model-a",)
    assert not hasattr(provider, "current_key_idx")
    assert clients.call_count == 2
    for call in clients.call_args_list:
        assert call.kwargs["http_options"].timeout == 30000
        assert call.kwargs["http_options"].retry_options.attempts == 1


@pytest.mark.parametrize("variable,value", [
    ("PROVIDER_TURN_TIMEOUT_SECONDS", "nan"), ("PROVIDER_CALL_TIMEOUT_SECONDS", "inf"),
    ("PROVIDER_MAX_ATTEMPTS", "0"), ("PROVIDER_MAX_ATTEMPTS_PER_CALL", "7"),
    ("PROVIDER_CALL_TIMEOUT_SECONDS", "0"), ("PROVIDER_MAX_ATTEMPTS", "not-a-number"),
])
def test_policy_rejects_invalid_operator_limits(monkeypatch, variable, value):
    monkeypatch.setenv(variable, value)
    with pytest.raises(ValueError):
        RouterPolicy.from_environment()


@pytest.mark.parametrize("code", [400, 401, 403, 404, 422, 501])
def test_permanent_api_errors_fail_once_and_redact_provider_details(clients, clock, code, caplog):
    provider = router()
    provider.clients[0].models.generate_content.side_effect = failure(code)
    with pytest.raises(ProviderRequestError) as raised:
        provider.models.generate_content(contents="question")
    assert raised.value.code == code
    assert "private-provider" not in str(raised.value) + caplog.text
    assert provider.clients[0].models.generate_content.call_count == 1
    provider.clients[1].models.generate_content.assert_not_called()
    assert clock.sleeps == []


@pytest.mark.parametrize("code", [408, 429, 500, 502, 503, 504])
def test_transient_rotation_is_sticky_within_request_and_resets_for_next(clients, clock, code):
    provider, trace = router(models=("model-a",)), ExecutionTrace()
    provider.clients[0].models.generate_content.side_effect = failure(code)
    provider.clients[1].models.generate_content.return_value = response()
    with provider_request(provider.policy, trace):
        provider.models.generate_content(contents="first")
        provider.models.generate_content(contents="second")
    assert provider.clients[0].models.generate_content.call_count == 1
    assert provider.clients[1].models.generate_content.call_count == 2
    assert trace.snapshot()["provider_attempts"] == 3
    assert trace.snapshot()["provider_retry_wait_seconds"] == 1
    provider.models.generate_content(contents="independent request")
    assert provider.clients[0].models.generate_content.call_count == 2
    assert provider.clients[1].models.generate_content.call_count == 3


def test_fallback_does_not_replace_explicit_model_in_a_different_phase(clients, clock):
    provider = router(keys=("key-a",), models=("primary", "fallback"))
    models = []

    def generate(**kwargs):
        models.append(kwargs["model"])
        if kwargs["model"] == "primary":
            raise failure(503)
        return response()

    provider.clients[0].models.generate_content.side_effect = generate
    with provider_request(provider.policy):
        provider.models.generate_content(model="primary", contents="answer")
        provider.models.generate_content(model="judge", contents="check")
    assert models == ["primary", "fallback", "judge"]


def test_default_failover_reaches_the_functionally_verified_fallback(clients, clock):
    provider = GeminiRouter(["key-a", "key-b"], policy=RouterPolicy())
    attempted = []

    def generate(**kwargs):
        attempted.append(kwargs["model"])
        if kwargs["model"] == "models/gemini-3.6-flash":
            raise failure(503)
        assert kwargs["model"] == "models/gemini-3.5-flash-lite"
        return response()

    for client in provider.clients:
        client.models.generate_content.side_effect = generate
    assert provider.models.generate_content(contents="question").text == "Complete answer"
    assert attempted == ["models/gemini-3.6-flash", "models/gemini-3.6-flash", "models/gemini-3.5-flash-lite"]


@pytest.mark.parametrize("hint", ["seconds", "date", "rpc"])
def test_respects_retry_guidance_and_preserves_input_config(clients, clock, hint):
    provider = router(models=("model-a",))
    raw = "3" if hint == "seconds" else format_datetime(datetime.fromtimestamp(1700000003, timezone.utc))
    details = [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "3s"}]
    provider.clients[0].models.generate_content.side_effect = failure(
        429, retry_after=raw if hint != "rpc" else None, details=details if hint == "rpc" else None)
    provider.clients[1].models.generate_content.return_value = response()
    config = types.GenerateContentConfig(temperature=0.3, http_options=types.HttpOptions(
        timeout=500, headers={"custom": "preserved"}, retry_options=types.HttpRetryOptions(attempts=5)))
    original = config.model_dump()
    provider.models.generate_content(contents="question", config=config)
    assert clock.sleeps == [3]
    assert config.model_dump() == original
    sent = provider.clients[1].models.generate_content.call_args.kwargs["config"]
    assert sent.http_options.timeout == 500
    assert sent.http_options.retry_options.attempts == 1
    assert sent.http_options.headers["custom"] == "preserved"
    assert sent.automatic_function_calling.disable is True


def test_hint_longer_than_remaining_budget_cannot_be_bypassed_by_another_phase(clients, clock):
    provider = router(RouterPolicy(turn_timeout_seconds=5))
    provider.clients[0].models.generate_content.side_effect = failure(429, retry_after="60")
    with provider_request(provider.policy):
        for model in ("optimiser", "answer"):
            with pytest.raises(ProviderRequestError) as raised:
                provider.models.generate_content(model=model, contents="question")
            assert raised.value.code == 429
    assert provider.clients[0].models.generate_content.call_count == 1
    assert clock.sleeps == []


@pytest.mark.parametrize("violation", [
    {"quotaId": "GenerateRequestsPerDayPerProjectPerModel"},
    {"quotaId": "GenerateRequestsPerMinute", "quotaValue": "0"},
])
def test_known_daily_or_zero_quota_fails_fast_across_phases(clients, clock, violation):
    provider = router()
    provider.clients[0].models.generate_content.side_effect = failure(429, details=[{
        "@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [violation]}])
    with provider_request(provider.policy):
        for model in ("optimiser", "answer"):
            with pytest.raises(ProviderRequestError):
                provider.models.generate_content(model=model, contents="question")
    assert provider.clients[0].models.generate_content.call_count == 1
    assert clock.sleeps == []


def test_per_call_and_total_attempt_caps_are_independent(clients, clock):
    provider = router(RouterPolicy(max_attempts=4, max_attempts_per_call=3), keys=("key-a",), models=("model-a",))
    provider.clients[0].models.generate_content.side_effect = failure(503)
    with provider_request(provider.policy):
        with pytest.raises(ProviderRequestError):
            provider.models.generate_content(contents="first")
        with pytest.raises(ProviderAttemptsExceeded):
            provider.models.generate_content(contents="second")
    assert provider.clients[0].models.generate_content.call_count == 4


def test_nested_scope_cannot_reset_attempt_budget(clients, clock):
    provider = router(RouterPolicy(max_attempts=1))
    provider.clients[0].models.generate_content.return_value = response()
    with provider_request(provider.policy) as outer:
        provider.models.generate_content(contents="first")
        with provider_request(RouterPolicy(max_attempts=100)) as inner:
            assert inner is outer
            with pytest.raises(ProviderAttemptsExceeded):
                provider.models.generate_content(contents="second")
    assert outer.attempts == 1


def test_invalid_local_config_does_not_consume_dispatch_budget(clients, clock):
    provider, trace = router(), ExecutionTrace()
    with provider_request(provider.policy, trace):
        with pytest.raises(ValueError):
            provider.models.generate_content(contents="question", config={"temperature": "invalid"})
    assert trace.snapshot()["provider_attempts"] == 0
    provider.clients[0].models.generate_content.assert_not_called()


def test_read_timeout_is_not_automatically_retried(clients, clock):
    provider = router()
    provider.clients[0].models.generate_content.side_effect = httpx.ReadTimeout("private timeout detail")
    with pytest.raises(ProviderTimeoutError) as raised:
        provider.models.generate_content(contents="question")
    assert "private timeout" not in str(raised.value)
    provider.clients[1].models.generate_content.assert_not_called()
    assert clock.sleeps == []


def test_connect_failure_can_retry_before_a_response_is_received(clients, clock):
    provider = router()
    provider.clients[0].models.generate_content.side_effect = httpx.ConnectError("private connect detail")
    provider.clients[1].models.generate_content.return_value = response()
    assert provider.models.generate_content(contents="question").text == "Complete answer"
    assert clock.sleeps == [1]


def test_late_response_is_rejected_and_no_further_dispatch_is_allowed(clients, clock):
    provider = router(RouterPolicy(turn_timeout_seconds=2))

    def late(**kwargs):
        clock.now += 3
        return response()

    provider.clients[0].models.generate_content.side_effect = late
    with provider_request(provider.policy):
        for _ in range(2):
            with pytest.raises(ProviderDeadlineExceeded):
                provider.models.generate_content(contents="question")
    assert provider.clients[0].models.generate_content.call_count == 1


def test_actual_graph_shares_budget_across_decomposition_generation_and_judge(clients, clock):
    provider, trace = router(RouterPolicy(turn_timeout_seconds=5, call_timeout_seconds=10, max_attempts=4)), ExecutionTrace()
    tool_call = types.GenerateContentResponse(candidates=[types.Candidate(finish_reason="STOP", content=types.Content(
        role="model", parts=[types.Part.from_function_call(name="search_knowledge_base", args={"query": "public"})]))])
    replies = iter([response('["focused query"]'), tool_call, response("Answer [1]."),
                    response('{"is_faithful":true,"reason":"Supported."}')])
    timeouts = []

    def generate(**kwargs):
        timeouts.append(kwargs["config"].http_options.timeout)
        clock.now += 1
        return next(replies)

    provider.clients[0].models.generate_content.side_effect = generate
    answer = run_agent_turn("Question", [], provider, build_agent_config(types.Tool(), {}),
                            {"search_knowledge_base": lambda query: "Public evidence"}, model="model-a",
                            use_decomposition=True, use_guardrail=True, execution=trace)
    assert "Completed" in answer
    assert timeouts == [5000, 4000, 3000, 2000]
    assert trace.snapshot()["provider_attempts"] == 4
    assert trace.snapshot()["generation_calls"] == 2
    assert trace.snapshot()["decomposition"] == "completed"
    assert trace.snapshot()["grounding"] == "completed"


def test_expired_decomposition_cannot_fall_back_into_more_provider_calls(clients, clock):
    provider, trace, history = router(RouterPolicy(turn_timeout_seconds=1)), ExecutionTrace(), []

    def late(**kwargs):
        clock.now += 2
        return response('["query"]')

    provider.clients[0].models.generate_content.side_effect = late
    with pytest.raises(ProviderDeadlineExceeded):
        run_agent_turn("Question", history, provider, build_agent_config(types.Tool(), {}), {},
                       use_decomposition=True, execution=trace)
    assert history == []
    assert trace.snapshot()["status"] == "timed_out"
    assert trace.snapshot()["provider_attempts"] == 1
    assert provider.clients[0].models.generate_content.call_count == 1


def test_auth_failure_cannot_be_retried_by_a_later_agent_phase(clients, clock):
    provider, trace = router(), ExecutionTrace()
    provider.clients[0].models.generate_content.side_effect = failure(401)
    with pytest.raises(ProviderRequestError):
        run_agent_turn("Question", [], provider, build_agent_config(types.Tool(), {}), {},
                       use_decomposition=True, execution=trace)
    assert trace.snapshot()["provider_attempts"] == 1
    assert provider.clients[0].models.generate_content.call_count == 1
    provider.clients[1].models.generate_content.assert_not_called()


def test_no_provider_dispatch_after_a_slow_retrieval_returns(clients, clock):
    provider, trace, history = router(RouterPolicy(turn_timeout_seconds=1)), ExecutionTrace(), []
    provider.clients[0].models.generate_content.return_value = types.GenerateContentResponse(candidates=[
        types.Candidate(finish_reason="STOP", content=types.Content(role="model", parts=[
            types.Part.from_function_call(name="search_knowledge_base", args={"query": "public"})]))])

    def slow_tool(query):
        clock.now += 2
        return "Late evidence"

    with pytest.raises(ProviderDeadlineExceeded):
        run_agent_turn("Question", history, provider, build_agent_config(types.Tool(), {}),
                       {"search_knowledge_base": slow_tool}, execution=trace)
    assert history == []
    assert trace.snapshot()["status"] == "timed_out"
    assert trace.snapshot()["provider_attempts"] == 1


def test_completed_answer_can_report_judge_unavailable_without_exceeding_attempt_limit(clients, clock):
    provider, trace = router(RouterPolicy(max_attempts=1)), ExecutionTrace()
    provider.clients[0].models.generate_content.return_value = response("Answer.")
    result = run_agent_turn("Question", [], provider, build_agent_config(types.Tool(), {}), {},
                           execution=trace, use_guardrail=True)
    assert "Unavailable" in result
    assert "Automated grounding check: Completed" not in result
    assert trace.snapshot()["provider_attempts"] == 1
    assert trace.snapshot()["grounding"] == "unavailable"


@pytest.mark.parametrize("error,status", [
    (ProviderDeadlineExceeded(), "timed_out"), (ProviderAttemptsExceeded(), "budget_exhausted"),
])
def test_web_budget_outcomes_preserve_history(error, status):
    client, trace = MagicMock(), ExecutionTrace()
    client.models.generate_content.side_effect = error
    previous = types.Content(role="model", parts=[types.Part(text="Previous completed answer")])
    chat = make_chat_fn(client, {}, "unused.json", None, ready_collection(), None, return_evidence=True)
    output = chat("Question", [], [previous], "Standard Vector Search", False, False, execution=trace)
    assert output[1] == [previous]
    assert output[5] is None
    assert trace.snapshot()["status"] == status
    assert "previous completed context" in output[0][-1]["content"].lower()


def test_tool_worker_context_keeps_the_same_budget(clients, clock):
    provider, trace = router(RouterPolicy(max_attempts=3)), ExecutionTrace()
    tool_call = types.GenerateContentResponse(candidates=[types.Candidate(finish_reason="STOP", content=types.Content(
        role="model", parts=[types.Part.from_function_call(name="search_knowledge_base", args={"query": "public"})]))])
    provider.clients[0].models.generate_content.side_effect = [tool_call, response("Nested tool evidence"), response("Answer.")]

    def tool(query):
        return provider.models.generate_content(model="model-a", contents=query).text

    assert run_agent_turn("Question", [], provider, build_agent_config(types.Tool(), {}),
                          {"search_knowledge_base": tool}, model="model-a", execution=trace) == "Answer."
    assert trace.snapshot()["provider_attempts"] == 3


def test_concurrent_requests_do_not_share_route_cursors(clients, clock):
    provider = router(models=("model-a",))
    barrier = threading.Barrier(2)
    followup = threading.Barrier(2)
    visited = []

    def primary(**kwargs):
        visited.append((kwargs["contents"], "primary"))
        if kwargs["contents"] in ("first", "second"):
            barrier.wait(timeout=5)
        if kwargs["contents"] == "first":
            raise failure(429)
        return response()

    def secondary(**kwargs):
        visited.append((kwargs["contents"], "secondary"))
        return response()

    provider.clients[0].models.generate_content.side_effect = primary
    provider.clients[1].models.generate_content.side_effect = secondary

    def request(question):
        trace = ExecutionTrace()
        with provider_request(provider.policy, trace):
            provider.models.generate_content(contents=question)
            followup.wait(timeout=5)
            provider.models.generate_content(contents=question + "-followup")
        return trace.snapshot()["provider_attempts"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(request, ("first", "second"))) == [3, 2]
    assert ("second", "secondary") not in visited
    assert ("second-followup", "primary") in visited
    assert ("second-followup", "secondary") not in visited
    assert ("first-followup", "secondary") in visited


def test_actual_sdk_uses_bounded_http_options_without_hidden_retries(clock):
    calls = []

    def handle(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "1"}, json={
                "error": {"code": 429, "message": "synthetic quota", "status": "RESOURCE_EXHAUSTED"}})
        return httpx.Response(200, json={"candidates": [{"finishReason": "STOP", "content": {
            "role": "model", "parts": [{"text": "Complete answer"}]}}]})

    original = genai.Client
    with httpx.Client(transport=httpx.MockTransport(handle)) as http:
        def client(**kwargs):
            kwargs["http_options"] = kwargs["http_options"].model_copy(update={"httpx_client": http})
            return original(**kwargs)

        with patch("llm_router.genai.Client", side_effect=client):
            provider = router(RouterPolicy(turn_timeout_seconds=5, call_timeout_seconds=3),
                              keys=("fixture-key",), models=("model-a",))
        config = types.GenerateContentConfig(http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=5)))
        assert provider.models.generate_content(contents="Public synthetic question", config=config).text == "Complete answer"
    assert len(calls) == 2
    assert all(request.extensions["timeout"]["read"] <= 3 for request in calls)
    assert all(request.headers["X-Server-Timeout"] == "3" for request in calls)
