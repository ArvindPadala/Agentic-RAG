"""Evidence provenance, concurrency, bounded history and authorization checks."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from unittest.mock import MagicMock, patch
import threading
import time

import gradio as gr
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from access_control import RuntimePolicy
from agent import build_search_tool
from app import build_ui, make_chat_fn
from evidence import (
    EvidenceCollector, EvidenceHistory, EvidenceTurn, MAX_RETAINED_TURNS,
    MAX_PASSAGES_PER_TURN, MAX_RETAINED_PASSAGES,
    authorized_passages, format_citations, render_cards,
)
from tests.test_runtime_privacy import collection  # noqa: F401 - real scoped Chroma fixture
from tests.test_supabase_memory import B, ORIGIN, sessions, snapshot
from private_sessions import COOKIE_NAME
from tests.test_showcase_corpus import document  # noqa: F401 - synthetic PDF fixture
from showcase_corpus import ShowcaseCorpus


@pytest.fixture(autouse=True)
def no_reranker_downloads():
    # These tests exercise provenance and lifecycle, not ranking quality.
    ranker = MagicMock()
    ranker.predict.side_effect = lambda pairs: [1.0] * len(pairs)
    with patch("hybrid_search.get_reranker", return_value=ranker):
        yield


def result(chunk_id="public", text="Public showcase passage", source="approved/fixture.pdf"):
    return {"chunk_id": chunk_id, "text": text, "source_document": "Fixture report (2023)",
            "source_pdf_key": source, "source_url": "https://www.ipcc.ch/fixture.pdf",
            "page": 0, "bbox": [0, 0, 1, 1], "score": 0.8, "chunk_type": "text"}


def test_parallel_repeated_searches_have_unique_stable_ids():
    collector = EvidenceCollector(first_number=7)
    with ThreadPoolExecutor(max_workers=5) as pool:
        records = list(pool.map(lambda index: collector.add(result(str(index % 3))), range(30)))
    assert {record.number for record in records} == {7, 8, 9}
    assert len(collector.snapshot()) == 3
    assert all(len({p.number for p in records if p.chunk_id == cid}) == 1 for cid in ("0", "1", "2"))


def test_missing_geometry_retains_text_without_inventing_a_highlight():
    record = EvidenceCollector().add({**result(), "bbox": [0, 0, 0, 0]})
    assert record.excerpt == "Public showcase passage"
    assert record.bbox == ()


def test_search_capture_is_scoped_and_defers_image_work(collection):
    collection.update(ids=["public"], metadatas=[{"source_pdf_key": "approved/fixture.pdf"}])
    for hybrid in (False, True):
        collector, renderer, s3 = EvidenceCollector(), MagicMock(), MagicMock()
        search, _ = build_search_tool(collection, None, s3, "fixture-bucket", use_hybrid=hybrid,
                                      citation_renderer=renderer, evidence_collector=collector)
        text = search("showcase")
        assert "Evidence [1]" in text
        assert "Public showcase passage" in text
        assert "Private sentinel" not in text
        assert "approved/fixture.pdf" not in text
        assert collector.snapshot()[0].source_key == "approved/fixture.pdf"
        renderer.assert_not_called()
        s3.head_object.assert_not_called()


def test_only_captured_refs_become_links_and_model_cannot_choose_destinations():
    record = EvidenceCollector().add(result())
    text, cited = format_citations("Claim [1](https://evil.invalid). Also [[1]](https://evil.invalid). Other claim [99].", [record])
    assert "[[1]](#citation-1)" in text
    assert "evil.invalid" not in text
    assert "[99]" not in text
    assert "source unavailable" in text
    assert cited == {1}


def test_cards_distinguish_retrieved_from_cited_and_escape_content():
    collector = EvidenceCollector()
    first = collector.add(result(text="<script>unsafe()</script> & an excerpt"))
    second = collector.add(result("other"))
    turn = EvidenceTurn("Question", collector.snapshot(), frozenset({first.number}), 1)
    cards = render_cards([turn], [first, second])
    assert "Cited in this answer" in cards
    assert "Retrieved evidence · not cited" in cards
    assert "not independently verified" in cards
    assert "<script>" not in cards
    assert "&lt;script&gt;" in cards
    assert "PDF page 1" in cards
    assert "#page=1" in cards
    assert first.source_key not in cards


def test_retained_evidence_observes_revocation_before_redisplay(collection):
    collection.update(ids=["public"], metadatas=[{"source_pdf_key": "approved/fixture.pdf"}])
    collector = EvidenceCollector()
    record = collector.add(result())
    assert authorized_passages([record], collection, RuntimePolicy()) == (record,)
    collection.update(ids=["public"], metadatas=[{"visibility": "private"}])
    assert authorized_passages([record], collection, RuntimePolicy()) == ()
    turn = EvidenceTurn("Question", (record,), frozenset({1}), 1)
    cards = render_cards([turn], ())
    assert "no longer available" in cards
    assert record.excerpt not in cards


def test_history_is_bounded_and_its_session_copy_has_an_independent_lock():
    history = EvidenceHistory()
    for index in range(MAX_RETAINED_TURNS + 3):
        record = EvidenceCollector(history.next_number).add(result(str(index)))
        history.append(EvidenceTurn("Question", (record,), frozenset({record.number})))
    assert len(history.turns) == MAX_RETAINED_TURNS
    assert history.next_number == MAX_RETAINED_TURNS + 4
    copied = deepcopy(history)
    assert copied.lock is not history.lock
    copied.clear()
    assert len(history.turns) == MAX_RETAINED_TURNS


def test_total_passage_cap_preserves_latest_answer_without_reusing_ids():
    history = EvidenceHistory()
    for turn_index in range(6):
        collector = EvidenceCollector(history.next_number)
        for passage_index in range(MAX_PASSAGES_PER_TURN + 1):
            collector.add(result(f"{turn_index}:{passage_index}"))
        assert len(collector.snapshot()) == MAX_PASSAGES_PER_TURN
        history.append(EvidenceTurn(f"Question {turn_index}", collector.snapshot(), frozenset()))
    assert len(history.records()) <= MAX_RETAINED_PASSAGES
    assert history.turns[-1].question == "Question 5"
    assert history.next_number == 6 * MAX_PASSAGES_PER_TURN + 1


def test_evicted_evidence_does_not_leave_untracked_provider_context():
    history, context = EvidenceHistory(), []
    for index in range(MAX_RETAINED_TURNS + 3):
        start = len(context)
        context.append(f"message {index}")
        record = EvidenceCollector(history.next_number).add(result(str(index)))
        history.append(EvidenceTurn("Question", (record,), frozenset(), context_start=start))
        context = history.trim_context(context)
    assert context == [f"message {index}" for index in range(3, MAX_RETAINED_TURNS + 3)]
    assert [turn.context_start for turn in history.turns] == list(range(MAX_RETAINED_TURNS))


def test_failed_turn_does_not_publish_captured_sources_or_damage_history(collection):
    chat = make_chat_fn(None, {}, "unused.json", None, collection, None, return_evidence=True)
    original = ["previous completed turn"]

    def fail_after_search(**kwargs):
        kwargs["tool_map"]["search_knowledge_base"]("showcase")
        kwargs["conversation_history"].append("unfinished tool cycle")
        raise RuntimeError("provider-private-sentinel")

    with patch("app.run_agent_turn", side_effect=fail_after_search):
        response = chat("Question", [], original, "Standard Vector Search", False, False)
    assert response[5] is None
    assert response[1] == original == ["previous completed turn"]
    assert "provider-private-sentinel" not in str(response)


def test_revocation_during_generation_withholds_prose_not_just_citations(collection):
    chat = make_chat_fn(None, {}, "unused.json", None, collection, None, return_evidence=True)

    def revoke(**kwargs):
        kwargs["tool_map"]["search_knowledge_base"]("showcase")
        kwargs["conversation_history"].append("unfinished revoked turn")
        collection.update(ids=["public"], metadatas=[{"visibility": "private"}])
        return "revoked-prose-sentinel [1]"

    with patch("app.run_agent_turn", side_effect=revoke):
        output = chat("Question", [], [], "Standard Vector Search", False, False)
    assert "revoked-prose-sentinel" not in str(output)
    assert output[1] == [] and output[5] is None
    assert "sources changed" in output[0][-1]["content"]


def test_revoked_history_is_not_sent_to_provider_on_followup(collection):
    demo = build_ui(None, {}, "unused.json", None, collection, None)
    app = gr.routes.App.create_app(demo)
    contexts = []

    def turn(**kwargs):
        contexts.append(list(kwargs["conversation_history"]))
        return tool_using_turn(**kwargs)

    try:
        with TestClient(app) as client, patch("app.run_agent_turn", side_effect=turn):
            client.post("/gradio_api/api/submit", json={
                "data": ["First question", [], None, None, None, None, None, None],
                "session_hash": "revoked-history"})
            collection.update(ids=["public"], metadatas=[{"visibility": "private"}])
            collection.add(ids=["replacement"], documents=["New approved showcase passage"],
                           metadatas=[{"visibility": "public", "source_document": "New report", "page": 0}])
            reply = client.post("/gradio_api/api/submit", json={
                "data": ["Follow-up", [], None, None, None, None, None, None],
                "session_hash": "revoked-history"})
            assert reply.status_code == 200
            assert "First question" not in reply.json()["data"][2]
        assert contexts == [[], []]
    finally:
        demo.close()


def tool_using_turn(**kwargs):
    text = kwargs["tool_map"]["search_knowledge_base"]("showcase")
    import re
    number = re.search(r"Evidence \[(\d+)\]", text).group(1)
    kwargs["conversation_history"].append("completed synthetic turn")
    return f"Source-grounded fixture answer [{number}]."


def test_capture_survives_omitted_citations_without_calling_renderer(collection):
    renderer = MagicMock()
    showcase = MagicMock(render_citation=renderer)
    chat = make_chat_fn(None, {}, "unused.json", None, collection, None,
                        showcase=showcase, return_evidence=True)

    def uncited(**kwargs):
        kwargs["tool_map"]["search_knowledge_base"]("showcase")
        return "Answer with no citation and no image URL."

    with patch("app.run_agent_turn", side_effect=uncited):
        output = chat("Question", [], [], "Standard Vector Search", False, False)
    assert len(output[5].records) == 1
    assert not output[5].cited
    assert output[5].records[0].excerpt == "Public showcase passage"
    renderer.assert_not_called()


def test_old_answer_selection_unknown_id_and_clear_are_request_scoped(collection):
    showcase = MagicMock()
    showcase.documents = {}
    showcase.render_citation.return_value = "/showcase/citations/" + "a" * 64 + ".png"
    demo = build_ui(None, {}, "unused.json", None, collection, None, showcase=showcase)
    app = gr.routes.App.create_app(demo)
    try:
        with TestClient(app) as client, patch("app.run_agent_turn", side_effect=tool_using_turn):
            for question in ("First question", "Second question"):
                reply = client.post("/gradio_api/api/submit", json={
                    "data": [question, [], None, None, None, None, None, None],
                    "session_hash": "evidence-session-A"})
                assert reply.status_code == 200
            cards = reply.json()["data"][2]
            assert "First question" in cards and "Second question" in cards
            assert "citation-1" in cards and "citation-2" in cards
            selected = client.post("/gradio_api/api/select_evidence", json={
                "data": [None, None], "event_data": {"citation": 1}, "session_hash": "evidence-session-A"})
            assert selected.status_code == 200
            assert "[1]" in selected.json()["data"][1]
            other_session = client.post("/gradio_api/api/select_evidence", json={
                "data": [None, None], "event_data": {"citation": 1}, "session_hash": "evidence-session-B"})
            assert other_session.status_code == 200
            assert "First question" not in other_session.text
            assert "unavailable" in other_session.json()["data"][1]
            unknown = client.post("/gradio_api/api/select_evidence", json={
                "data": [None, None], "event_data": {"citation": 999}, "session_hash": "evidence-session-A"})
            assert "unavailable" in unknown.json()["data"][1]
            client.post("/gradio_api/api/clear_chat", json={
                "data": [None, None], "session_hash": "evidence-session-A"})
            cleared = client.post("/gradio_api/api/select_evidence", json={
                "data": [None, None], "event_data": {"citation": 1}, "session_hash": "evidence-session-A"})
            assert "First question" not in cleared.text
            assert "unavailable" in cleared.json()["data"][1]
    finally:
        demo.close()


def test_followup_preserves_context_and_selected_settings(collection):
    demo = build_ui(None, {}, "unused.json", None, collection, None)
    app = gr.routes.App.create_app(demo)
    received = []

    def turn(**kwargs):
        received.append((list(kwargs["conversation_history"]), kwargs["use_decomposition"], kwargs["use_guardrail"]))
        return tool_using_turn(**kwargs)

    try:
        with TestClient(app) as client, patch("app.run_agent_turn", side_effect=turn):
            preset = client.post("/gradio_api/api/example_compare", json={
                "data": [], "session_hash": "followup-test"}).json()["data"][0]
            client.post("/gradio_api/api/submit", json={
                "data": [preset, [], None, None, None, None, None, None], "session_hash": "followup-test"})
            followup = client.post("/gradio_api/api/followup_explain", json={
                "data": [], "session_hash": "followup-test"}).json()["data"][0]
            client.post("/gradio_api/api/submit", json={
                "data": [followup, [], None, None, None, None, None, None], "session_hash": "followup-test"})
        assert received[0] == ([], True, False)
        assert received[1] == (["completed synthetic turn"], True, False)
    finally:
        demo.close()


def test_opaque_page_route_hides_paths_and_rechecks_revocation(collection, document, tmp_path):
    source, path, digest = document
    corpus = ShowcaseCorpus(tmp_path, [document])
    key = f"showcase/{digest}/{source['id']}.pdf"
    collection.update(ids=["public"], metadatas=[{"source_pdf_key": key}])
    record = result(source=key)
    url = corpus.render_citation(record, opaque_url=True)
    assert str(tmp_path) not in url
    with TestClient(FastAPI(routes=corpus.routes(collection, RuntimePolicy()))) as client:
        response = client.get(url)
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        assert response.content.startswith(b"\x89PNG")
        collection.update(ids=["public"], metadatas=[{"visibility": "private"}])
        assert client.get(url).status_code == 404
        assert client.get("/showcase/citations/" + "b" * 64 + ".png").status_code == 404


def test_private_preview_in_flight_cannot_repopulate_ended_session(collection):
    backend = MagicMock()
    backend.start_guest.return_value = ("token-A", int(time.time() + 3600))
    backend.load_memory.return_value = snapshot()
    service = sessions(backend)
    showcase = MagicMock()
    showcase.documents = {}
    entered, release = threading.Event(), threading.Event()

    def delayed_render(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return "late-private-preview-sentinel"

    showcase.render_citation.side_effect = delayed_render
    demo = build_ui(None, {}, "unused.json", None, collection, None,
                    private_sessions=service, showcase=showcase)
    app = gr.routes.App.create_app(demo, app_kwargs={"routes": service.routes()})
    try:
        with TestClient(app, base_url=ORIGIN) as client, patch("app.run_agent_turn", side_effect=tool_using_turn), \
                ThreadPoolExecutor(max_workers=1) as executor:
            headers = {"Origin": ORIGIN}
            client.post("/private-session/start", headers=headers, json={"captcha_token": "fixture"})
            client.post("/gradio_api/api/refresh_private_session", headers=headers, json={
                "data": [[], None, None, None], "session_hash": "preview-lifecycle"})
            client.post("/gradio_api/api/submit", headers=headers, json={
                "data": ["private-question-sentinel", [], None, None, None, None, None, None],
                "session_hash": "preview-lifecycle"})
            pending = executor.submit(client.post, "/gradio_api/api/select_evidence", headers=headers, json={
                "data": [None, None], "event_data": {"citation": 1}, "session_hash": "preview-lifecycle"})
            try:
                assert entered.wait(10)
                client.post("/private-session/end", headers=headers)
                ended = client.post("/gradio_api/api/refresh_private_session", headers=headers, json={
                    "data": [[], None, None, None], "session_hash": "preview-lifecycle"})
                assert "private-question-sentinel" not in ended.text
            finally:
                release.set()
            finished = pending.result(timeout=10)
            assert "late-private-preview-sentinel" not in finished.text
            assert "private-question-sentinel" not in finished.text
    finally:
        release.set()
        demo.close()


def test_changed_private_identity_cannot_read_retained_evidence(collection):
    backend = MagicMock()
    backend.start_guest.return_value = ("token-A", int(time.time() + 3600))
    backend.load_memory.return_value = snapshot()
    service = sessions(backend)
    renderer = MagicMock()
    showcase = MagicMock(render_citation=renderer)
    showcase.documents = {}
    demo = build_ui(None, {}, "unused.json", None, collection, None,
                    private_sessions=service, showcase=showcase)
    app = gr.routes.App.create_app(demo, app_kwargs={"routes": service.routes()})
    try:
        with TestClient(app, base_url=ORIGIN) as client, patch("app.run_agent_turn", side_effect=tool_using_turn):
            headers = {"Origin": ORIGIN}
            client.post("/private-session/start", headers=headers, json={"captcha_token": "fixture"})
            client.post("/gradio_api/api/refresh_private_session", headers=headers, json={
                "data": [[], None, None, None], "session_hash": "identity-evidence"})
            client.post("/gradio_api/api/submit", headers=headers, json={
                "data": ["A-private-question-sentinel", [], None, None, None, None, None, None],
                "session_hash": "identity-evidence"})
            backend.load_memory.return_value = snapshot(B, "B-private-memory", "token-B")
            client.cookies.set(COOKIE_NAME, service._cookie("token-B", int(time.time() + 3600)),
                               domain="demo.example.invalid", path="/")
            denied = client.post("/gradio_api/api/select_evidence", headers=headers, json={
                "data": [None, None], "event_data": {"citation": 1}, "session_hash": "identity-evidence"})
            assert denied.status_code == 200
            assert "session changed" in denied.text
            assert "A-private-question-sentinel" not in denied.text
            renderer.assert_not_called()
    finally:
        demo.close()
