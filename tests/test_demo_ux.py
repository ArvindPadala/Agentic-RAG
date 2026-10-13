"""Visitor-flow regressions using authorized synthetic data, not real providers."""

from unittest.mock import MagicMock, patch

import pytest
import gradio as gr
from fastapi.testclient import TestClient

from access_control import RuntimePolicy
from app import DEFAULT_SEARCH_LABEL, build_ui, corpus_status, make_chat_fn, report_cards
from tests.test_runtime_privacy import collection, ready_collection  # noqa: F401 - shared real Chroma fixture


def test_readiness_uses_public_scope_and_observes_revocation(collection):
    policy = RuntimePolicy()
    assert corpus_status(collection, policy) == ("ready", 1)
    collection.update(ids=["public"], metadatas=[{"visibility": "private"}])
    assert collection.count() == 3
    assert corpus_status(collection, policy) == ("empty", 0)
    chat = make_chat_fn(None, {}, "unused.json", None, collection, None, policy=policy)
    with patch("app.build_search_tool") as tool, patch("app.run_agent_turn") as turn:
        result = chat("Question", [], ["completed turn"], "Standard Vector Search", True, True)
    assert "No public reports are ready" in result[0][-1]["content"]
    assert result[1] == ["completed turn"]
    tool.assert_not_called()
    turn.assert_not_called()


def test_availability_error_is_generic_and_skips_provider():
    coll = MagicMock()
    coll.get.side_effect = RuntimeError("private-internal-error")
    chat = make_chat_fn(None, {}, "unused.json", None, coll, None)
    with patch("app.run_agent_turn") as turn:
        result = chat("Question", [], [], "Standard Vector Search", False, False)
    assert "temporarily unavailable" in result[0][-1]["content"]
    assert "private-internal-error" not in str(result)
    turn.assert_not_called()


def test_report_cards_do_not_claim_private_source_is_searchable(collection):
    source = {"title": "Fixture report", "url": "https://www.ipcc.ch/fixture.pdf", "year": 2023}
    collection.update(ids=["private"], metadatas=[{"source_pdf_key": "fixture/private.pdf"}])
    showcase = MagicMock()
    showcase.documents = {"fixture/private.pdf": (None, source)}
    cards = report_cards(collection, RuntimePolicy(), showcase)
    assert len(cards) == 1
    assert "Not currently searchable" in cards[0]
    assert "Available for research" not in cards[0]
    assert "Private sentinel" not in cards[0]


@pytest.mark.parametrize("task,decomposition,check", [
    ("fact", False, False), ("compare", True, False),
    ("pathway", False, False), ("limits", False, True),
])
def test_actual_presets_set_execution_options_and_clear_restores_welcome(task, decomposition, check):
    demo = build_ui(None, {}, "unused.json", None, ready_collection(), None)
    app = gr.routes.App.create_app(demo)

    def turn(**kwargs):
        kwargs["conversation_history"].append("synthetic completed turn")
        return "Synthetic answer"

    try:
        with TestClient(app) as client, patch("app.run_agent_turn", side_effect=turn) as agent:
            response = client.post("/gradio_api/api/example_" + task,
                                   json={"data": [], "session_hash": "preset-test"})
            assert response.status_code == 200
            values = response.json()["data"]
            assert values[1:4] == [DEFAULT_SEARCH_LABEL, decomposition, check]
            result = client.post("/gradio_api/api/submit", json={
                "data": [values[0], [], None, None, None, None, None, None],
                "session_hash": "preset-test",
            })
            assert result.status_code == 200
            assert agent.call_args.kwargs["use_decomposition"] is decomposition
            assert agent.call_args.kwargs["use_guardrail"] is check
            assert result.json()["data"][5]["visible"] is False
            cleared = client.post("/gradio_api/api/clear_chat", json={
                "data": [None, None], "session_hash": "preset-test"})
            assert cleared.status_code == 200
            assert cleared.json()["data"][0]["value"] == []
            assert cleared.json()["data"][0]["visible"] is False
            assert cleared.json()["data"][5]["visible"] is True
    finally:
        demo.close()


def test_unavailable_controls_can_recover_without_reloading_or_generating():
    coll = ready_collection()
    coll.get.side_effect = RuntimeError("unavailable")
    demo = build_ui(None, {}, "unused.json", None, coll, None)
    components = {c["props"].get("elem_id"): c["props"] for c in demo.config["components"]}
    assert components["send-btn"]["interactive"] is False
    assert components["research-question"]["interactive"] is False
    coll.get.side_effect = None
    app = gr.routes.App.create_app(demo)
    try:
        with TestClient(app) as client, patch("app.run_agent_turn") as agent:
            result = client.post("/gradio_api/api/refresh_availability", json={"data": []})
            assert result.status_code == 200
            assert "Reports ready" in result.json()["data"][0]
            assert all(update["interactive"] for update in result.json()["data"][1:])
            agent.assert_not_called()
    finally:
        demo.close()
