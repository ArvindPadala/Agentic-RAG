"""Synthetic Auth/REST responses; no live Supabase, credentials, or LLM calls."""

import json
import base64
import time
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import gradio as gr
import pytest
import requests
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient
from starlette.requests import Request

from app import build_ui, make_chat_fn, make_save_fn
from config import Config
from private_sessions import COOKIE_NAME, PrivateSessions
from supabase_backend import (
    AuthenticationError, MemoryConflict, MemorySnapshot, SupabaseBackend,
    SupabaseError, empty_memory,
)


A = "11111111-1111-1111-1111-111111111111"
B = "22222222-2222-2222-2222-222222222222"
ORIGIN = "https://demo.example.invalid"


def response(data=None, status=200):
    if isinstance(data, list):
        for row in data:
            if isinstance(row, dict) and "memory" in row:
                row.setdefault("is_closed", False)
    result = MagicMock()
    result.status_code = status
    result.json.return_value = data
    return result


@pytest.fixture
def backend():
    return SupabaseBackend("https://fixture.supabase.co", "sb_publishable_fixture")


def memory(fact):
    return {"session_summaries": [], "preferences": {}, "facts": [fact]}


def snapshot(user_id=A, fact="A-private-sentinel", token="token-A", revision=1):
    return MemorySnapshot(user_id, revision, memory(fact), token)


def sessions(backend):
    return PrivateSessions(backend, Fernet.generate_key(), ORIGIN, "public-site-key")


def browser_request(service, token="token-A", expires=None):
    cookie = service._cookie(token, expires or int(time.time() + 3600))
    return Request({
        "type": "http", "headers": [
            (b"origin", ORIGIN.encode()),
            (b"cookie", (COOKIE_NAME + "=" + cookie).encode()),
        ],
    })


def test_identity_comes_from_auth_not_metadata_and_read_is_owner_filtered(backend):
    with patch("supabase_backend.requests.request", side_effect=[
        response({"id": A, "user_metadata": {"user_id": B, "role": "admin"}}),
        response([{"user_id": A, "revision": 1, "memory": memory("A")}]),
    ]) as http:
        loaded = backend.load_memory("token-A")
    assert loaded.user_id == A
    assert http.call_args.kwargs["params"]["user_id"] == "eq." + A
    assert http.call_args.kwargs["headers"]["Authorization"] == "Bearer token-A"
    assert http.call_args.kwargs["headers"]["apikey"] == "sb_publishable_fixture"
    assert http.call_args.kwargs["allow_redirects"] is False
    assert http.call_args.kwargs["timeout"] == (3, 10)
    assert "token-A" not in repr(loaded)


def test_backend_rejects_a_different_owners_record(backend):
    with patch("supabase_backend.requests.request", side_effect=[
        response({"id": A}),
        response([{"user_id": B, "revision": 1, "memory": memory("B-private-sentinel")}]),
    ]):
        with pytest.raises(SupabaseError, match="invalid record"):
            backend.load_memory("token-A")


def test_no_row_returns_fresh_empty_memory(backend):
    with patch("supabase_backend.requests.request", side_effect=[response({"id": A}), response([])]):
        loaded = backend.load_memory("token-A")
    assert loaded.revision == 0
    assert loaded.memory == empty_memory()


def test_rpc_write_rechecks_identity_and_does_not_accept_owner_in_payload(backend):
    with patch("supabase_backend.requests.request", side_effect=[
        response({"id": A}),
        response([{"user_id": A, "revision": 2, "memory": memory("updated")}]),
    ]) as http:
        saved = backend.save_memory(snapshot(), memory("updated"))
    assert saved.revision == 2
    assert http.call_args.kwargs["json"] == {
        "expected_revision": 1, "new_memory": memory("updated"),
    }
    with patch("supabase_backend.requests.request", return_value=response({"id": B})) as http:
        with pytest.raises(AuthenticationError):
            backend.save_memory(snapshot(), memory("updated"))
        assert http.call_count == 1


def test_conflict_never_retries_non_idempotent_write_or_exposes_response(backend):
    with patch("supabase_backend.requests.request", side_effect=[
        response({"id": A}), response({"message": "private-provider-detail"}, 409),
    ]) as http:
        with pytest.raises(MemoryConflict) as error:
            backend.save_memory(snapshot(), memory("updated"))
    assert "private-provider-detail" not in str(error.value)
    assert http.call_count == 2


@pytest.mark.parametrize("status,error", [(401, AuthenticationError), (429, SupabaseError), (503, SupabaseError), (302, SupabaseError)])
def test_auth_failure_and_outage_fail_closed_without_returning_provider_details(backend, status, error):
    with patch("supabase_backend.requests.request", return_value=response({"detail": "secret-sentinel"}, status)):
        with pytest.raises(error) as caught:
            backend.load_memory("token-A")
    assert "secret-sentinel" not in str(caught.value)


def test_network_error_is_sanitized_and_not_retried(backend):
    with patch("supabase_backend.requests.request", side_effect=requests.Timeout("token-A secret")) as http:
        with pytest.raises(SupabaseError) as caught:
            backend.load_memory("token-A")
        assert http.call_count == 1
    assert "token-A" not in str(caught.value)


def test_invalid_memory_is_rejected_before_any_network_write(backend):
    with patch("supabase_backend.requests.request") as http:
        with pytest.raises(SupabaseError):
            backend.save_memory(snapshot(), {"facts": ["missing other keys"]})
        with pytest.raises(SupabaseError):
            backend.save_memory(snapshot(), memory("x" * 20001))
        http.assert_not_called()


def test_guest_creation_requires_captcha_verifies_identity_and_discards_refresh_token(backend):
    with patch("supabase_backend.requests.request") as http:
        with pytest.raises(AuthenticationError):
            backend.start_guest("")
        http.assert_not_called()
    with patch("supabase_backend.requests.request", side_effect=[
        response({"access_token": "guest-token", "expires_in": 7200, "refresh_token": "discard-me"}),
        response({"id": A, "is_anonymous": True}),
    ]) as http:
        token, expires = backend.start_guest("captcha-fixture")
    assert token == "guest-token"
    assert expires <= time.time() + 3600
    assert http.call_args_list[0].kwargs["json"]["gotrue_meta_security"]["captcha_token"] == "captcha-fixture"


@pytest.mark.parametrize("key", ["sb_secret_fixture", "invalid", ""])
def test_privileged_or_invalid_key_is_not_accepted(key):
    with pytest.raises(ValueError):
        SupabaseBackend("https://fixture.supabase.co", key)


def test_cookie_tampering_expiry_and_cross_origin_fail_before_backend_access():
    backend = MagicMock()
    service = sessions(backend)
    request = browser_request(service, expires=int(time.time() - 60))
    with pytest.raises(AuthenticationError):
        service.load_for_request(request)
    request = Request({"type": "http", "headers": [(b"origin", ORIGIN.encode()),
                       (b"cookie", (COOKIE_NAME + "=tampered").encode())]})
    with pytest.raises(AuthenticationError):
        service.load_for_request(request)
    request = browser_request(service)
    request.scope["headers"][0] = (b"origin", b"https://attacker.example.invalid")
    with pytest.raises(AuthenticationError):
        service.load_for_request(request)
    backend.load_memory.assert_not_called()


def test_ordinary_public_chat_does_not_create_auth_users_or_read_memory():
    backend = MagicMock()
    service = sessions(backend)
    chat = make_chat_fn(None, {}, "never-read.json", None, MagicMock(), None,
                        private_sessions=service)
    with patch("app.run_agent_turn", return_value="Public answer"):
        result = chat("question", [], [], "Standard Vector Search", False, False)
    assert result[4] is None
    assert not backend.mock_calls


def test_two_guest_memories_are_isolated_and_identity_switch_clears_history():
    backend = MagicMock()
    backend.load_memory.side_effect = [snapshot(), snapshot(B, "B-private-sentinel", "token-B")]
    service = sessions(backend)
    chat = make_chat_fn(None, {}, "never-read.json", None, MagicMock(), None,
                        private_sessions=service)
    received = []

    def turn(**kwargs):
        received.append((kwargs["generation_config"].system_instruction,
                         list(kwargs["conversation_history"])))
        return "Private answer"

    with patch("app.run_agent_turn", side_effect=turn):
        a = chat("A question", [], [], "Standard Vector Search", False, False,
                 owner_id=A, request=browser_request(service))
        b = chat("B question", a[0], ["A private previous conversation"],
                 "Standard Vector Search", False, False,
                 owner_id=A, request=browser_request(service, "token-B"))
    assert "A-private-sentinel" in received[0][0]
    assert "B-private-sentinel" not in received[0][0]
    assert "B-private-sentinel" in received[1][0]
    assert "A-private-sentinel" not in received[1][0]
    assert received[1][1] == []
    assert a[4] == A and b[4] == B
    assert "A question" not in str(b)


def test_expired_private_identity_cannot_use_old_prompt_or_history():
    service = sessions(MagicMock())
    chat = make_chat_fn(None, {}, "never-read.json", None, MagicMock(), None,
                        private_sessions=service)
    with patch("app.run_agent_turn") as llm:
        result = chat("question", [{"role": "assistant", "content": "private"}], ["private"],
                      "Standard Vector Search", False, False, owner_id=A,
                      request=browser_request(service, expires=int(time.time() - 60)))
    assert result[0] == result[1] == []
    assert result[4] is None
    llm.assert_not_called()


def test_authenticated_save_uses_own_snapshot_and_never_local_file():
    backend = MagicMock()
    original = snapshot()
    backend.load_memory.return_value = original
    service = sessions(backend)
    save = make_save_fn(None, {"facts": ["legacy"]}, "never-written.json",
                        private_sessions=service)
    with patch("app.update_memory_from_conversation", return_value=memory("updated")) as extract, \
            patch("app.save_memory") as write:
        status = save(["synthetic conversation"], owner_id=A, request=browser_request(service))
    assert "saved" in status
    extract.assert_called_once()
    assert extract.call_args.kwargs["raise_on_error"] is True
    assert backend.save_memory.call_args.args == (original, memory("updated"))
    assert original.memory == memory("A-private-sentinel")
    write.assert_not_called()


def test_save_rejects_owner_switch_and_reports_extraction_or_conflict_failures():
    backend = MagicMock()
    backend.load_memory.return_value = snapshot()
    service = sessions(backend)
    save = make_save_fn(None, {}, "never-written.json", private_sessions=service)
    with patch("app.update_memory_from_conversation") as extract:
        assert "session changed" in save(["conversation"], owner_id=B, request=browser_request(service))
        extract.assert_not_called()
    with patch("app.update_memory_from_conversation", side_effect=RuntimeError("secret")):
        assert "could not" in save(["conversation"], owner_id=A, request=browser_request(service))
        backend.save_memory.assert_not_called()
    backend.save_memory.side_effect = MemoryConflict("Memory changed. Retry saving.")
    with patch("app.update_memory_from_conversation", return_value=empty_memory()):
        assert "Retry saving" in save(["conversation"], owner_id=A, request=browser_request(service))


def test_real_gradio_routes_encrypt_cookie_and_protect_start_and_forget():
    backend = MagicMock()
    backend.start_guest.return_value = ("raw-access-token-sentinel", int(time.time() + 3600))
    service = sessions(backend)
    demo = build_ui(None, {}, "never-read.json", None, MagicMock(), None,
                    private_sessions=service)
    app = gr.routes.App.create_app(demo, app_kwargs={"routes": service.routes()})
    try:
        with TestClient(app, base_url=ORIGIN) as client:
            blocked = client.post("/private-session/start", json={"captcha_token": "fixture"})
            assert blocked.status_code == 401
            backend.start_guest.assert_not_called()
            started = client.post("/private-session/start", headers={"Origin": ORIGIN},
                                  json={"captcha_token": "fixture"})
            assert started.status_code == 200
            cookie = started.headers["set-cookie"]
            assert "HttpOnly" in cookie and "Secure" in cookie and "SameSite=lax" in cookie
            assert "raw-access-token-sentinel" not in cookie + started.text + str(demo.config)
            blocked = client.post("/private-session/forget", headers={"Origin": "https://attacker.example.invalid"})
            assert blocked.status_code == 403
            assert "set-cookie" not in blocked.headers
            backend.forget_memory.assert_not_called()
            ended = client.post("/private-session/forget", headers={"Origin": ORIGIN}, json={"user_id": B})
            assert ended.status_code == 200
            backend.forget_memory.assert_called_once_with("raw-access-token-sentinel")
    finally:
        demo.close()


def test_oversized_guest_route_request_cannot_create_accounts():
    backend = MagicMock()
    service = sessions(backend)
    from fastapi import FastAPI
    with TestClient(FastAPI(routes=service.routes()), base_url=ORIGIN) as client:
        result = client.post("/private-session/start", headers={"Origin": ORIGIN},
                             content=json.dumps({"captcha_token": "x" * 10000}))
    assert result.status_code == 413
    backend.start_guest.assert_not_called()


def test_closed_identity_is_denied_even_while_auth_jwt_is_valid(backend):
    with patch("supabase_backend.requests.request", side_effect=[
        response({"id": A}),
        response([{"user_id": A, "revision": 2, "memory": empty_memory(), "is_closed": True}]),
    ]):
        with pytest.raises(AuthenticationError, match="closed"):
            backend.load_memory("still-valid-auth-token")


def test_forget_rpc_derives_owner_and_returns_confirmed_identity(backend):
    with patch("supabase_backend.requests.request", side_effect=[response({"id": A}), response(A)]) as http:
        backend.forget_memory("token-A")
    assert http.call_args.args[0] == "POST"
    assert http.call_args.args[1].endswith("/rpc/forget_user_memory")
    assert http.call_args.kwargs["json"] == {}
    with patch("supabase_backend.requests.request", side_effect=[response({"id": A}), response(B)]):
        with pytest.raises(SupabaseError, match="could not be cleared"):
            backend.forget_memory("token-A")


def test_legacy_anon_key_is_supported_but_service_role_is_rejected():
    def key(role):
        payload = base64.urlsafe_b64encode(json.dumps({"role": role}).encode()).decode().rstrip("=")
        return "header." + payload + ".signature"
    SupabaseBackend("https://fixture.supabase.co", key("anon"))
    with pytest.raises(ValueError):
        SupabaseBackend("https://fixture.supabase.co", key("service_role"))


def test_repeated_start_reuses_active_identity_instead_of_creating_another_user():
    from fastapi import FastAPI

    backend = MagicMock()
    backend.start_guest.return_value = ("token-A", int(time.time() + 3600))
    backend.load_memory.return_value = snapshot()
    service = sessions(backend)
    with TestClient(FastAPI(routes=service.routes()), base_url=ORIGIN) as client:
        for _ in range(2):
            result = client.post("/private-session/start", headers={"Origin": ORIGIN},
                                 json={"captcha_token": "fixture"})
            assert result.status_code == 200
    backend.start_guest.assert_called_once()
    backend.load_memory.assert_called_once_with("token-A")


def test_enabled_supabase_configuration_cannot_silently_fall_back_to_shared_memory():
    with patch.dict(os.environ, {"SUPABASE_ENABLED": "true", "APP_MODE": "public_demo"}):
        with patch.dict(os.environ):
            for key in ("SUPABASE_URL", "SUPABASE_PUBLISHABLE_KEY", "SESSION_COOKIE_KEY",
                        "APP_PUBLIC_URL", "TURNSTILE_SITE_KEY"):
                os.environ.pop(key, None)
            with pytest.raises(ValueError, match="SUPABASE_URL"):
                Config()
    with patch.dict(os.environ, {"SUPABASE_ENABLED": "true", "APP_MODE": "local_private"}):
        with pytest.raises(ValueError, match="public_demo"):
            Config()


def test_actual_gradio_save_callback_gets_cookie_from_request_not_ui_inputs():
    backend = MagicMock()
    backend.start_guest.return_value = ("token-A", int(time.time() + 3600))
    backend.load_memory.return_value = snapshot()
    service = sessions(backend)
    demo = build_ui(None, {}, "never-read.json", None, MagicMock(), None,
                    private_sessions=service)
    app = gr.routes.App.create_app(demo, app_kwargs={"routes": service.routes()})
    try:
        with TestClient(app, base_url=ORIGIN) as client:
            client.post("/private-session/start", headers={"Origin": ORIGIN},
                        json={"captcha_token": "fixture"})
            result = client.post("/gradio_api/api/save", headers={"Origin": ORIGIN},
                                 json={"data": [None, None, None], "session_hash": "synthetic-session"})
            assert result.status_code == 200
            assert "session changed" in result.json()["data"][0]
            backend.load_memory.assert_called_with("token-A")
            backend.save_memory.assert_not_called()
    finally:
        demo.close()


def test_return_to_public_mode_needs_no_storage_and_never_claims_deletion():
    from fastapi import FastAPI

    backend = MagicMock()
    service = sessions(backend)
    with TestClient(FastAPI(routes=service.routes()), base_url=ORIGIN) as client:
        client.cookies.set(COOKIE_NAME, service._cookie("token-A", int(time.time() + 3600)),
                           domain="demo.example.invalid", path="/")
        blocked = client.post("/private-session/end", headers={"Origin": "https://attacker.example.invalid"})
        assert blocked.status_code == 403 and "set-cookie" not in blocked.headers
        result = client.post("/private-session/end", headers={"Origin": ORIGIN})
        assert result.json() == {"status": "public", "memory_deleted": False}
        assert COOKIE_NAME not in client.cookies
        assert not backend.mock_calls


def test_late_private_response_is_discarded_after_return_to_public_mode():
    backend = MagicMock()
    backend.start_guest.return_value = ("token-A", int(time.time() + 3600))
    backend.load_memory.return_value = snapshot()
    service = sessions(backend)
    demo = build_ui(None, {}, "never-read.json", None, MagicMock(), None,
                    private_sessions=service)
    app = gr.routes.App.create_app(demo, app_kwargs={"routes": service.routes()})
    entered, release = threading.Event(), threading.Event()

    def delayed_turn(**kwargs):
        entered.set()
        assert release.wait(10)
        return "late-private-answer-sentinel"

    try:
        with TestClient(app, base_url=ORIGIN) as client, \
                patch("app.run_agent_turn", side_effect=delayed_turn), \
                ThreadPoolExecutor(max_workers=1) as executor:
            headers = {"Origin": ORIGIN}
            client.post("/private-session/start", headers=headers, json={"captcha_token": "fixture"})
            initialized = client.post("/gradio_api/api/refresh_private_session", headers=headers,
                                      json={"data": [[], None, None, None], "session_hash": "lifecycle-test"})
            assert initialized.status_code == 200
            pending = executor.submit(client.post, "/gradio_api/api/submit", headers=headers, json={
                "data": ["question", [], None, None, None, None, None, None],
                "session_hash": "lifecycle-test",
            })
            try:
                assert entered.wait(10)
                assert client.post("/private-session/end", headers=headers).status_code == 200
                ended = client.post("/gradio_api/api/refresh_private_session", headers=headers,
                                    json={"data": [[], None, None, None], "session_hash": "lifecycle-test"})
                assert ended.status_code == 200
            finally:
                release.set()
            completed = pending.result(timeout=10)
            assert completed.status_code == 200
            assert "late-private-answer-sentinel" not in completed.text
            assert "A-private-sentinel" not in completed.text
    finally:
        release.set()
        demo.close()


def test_lifecycle_change_during_extraction_cancels_private_write():
    backend = MagicMock()
    backend.load_memory.return_value = snapshot()
    service = sessions(backend)
    scope = {"generation": "before"}
    save = make_save_fn(None, {}, "never-written.json", private_sessions=service)

    def extraction(*args, **kwargs):
        scope["generation"] = "after"
        return memory("new")

    with patch("app.update_memory_from_conversation", side_effect=extraction):
        result = save(["conversation"], owner_id=A, scope_state=scope, request=browser_request(service))
    assert result == gr.skip()
    backend.save_memory.assert_not_called()


def test_late_memory_refresh_cannot_restore_private_panel_after_ending_session():
    backend = MagicMock()
    backend.start_guest.return_value = ("token-A", int(time.time() + 3600))
    original = snapshot()
    original.memory["session_summaries"] = ["refresh-private-summary-sentinel"]
    entered, release = threading.Event(), threading.Event()

    def delayed_load(token):
        entered.set()
        assert release.wait(10)
        return original

    backend.load_memory.side_effect = delayed_load
    service = sessions(backend)
    demo = build_ui(None, {}, "never-read.json", None, MagicMock(), None,
                    private_sessions=service)
    app = gr.routes.App.create_app(demo, app_kwargs={"routes": service.routes()})
    try:
        with TestClient(app, base_url=ORIGIN) as client, ThreadPoolExecutor(max_workers=1) as executor:
            headers = {"Origin": ORIGIN}
            client.post("/private-session/start", headers=headers, json={"captcha_token": "fixture"})
            payload = {"data": [[], None, None, None], "session_hash": "refresh-lifecycle-test"}
            pending = executor.submit(client.post, "/gradio_api/api/refresh_private_session",
                                      headers=headers, json=payload)
            try:
                assert entered.wait(10)
                client.post("/private-session/end", headers=headers)
                latest = client.post("/gradio_api/api/refresh_private_session", headers=headers, json=payload)
                assert latest.status_code == 200
            finally:
                release.set()
            completed = pending.result(timeout=10)
            assert completed.status_code == 200
            assert "refresh-private-summary-sentinel" not in completed.text
    finally:
        release.set()
        demo.close()


def confirmed_memory():
    return {
        "session_summaries": ["confirmed-summary-sentinel"],
        "preferences": {"answer_style": "fixture preference"},
        "facts": ["confirmed-fact"],
    }


def test_save_panel_uses_confirmed_record_not_extraction_proposal():
    backend = MagicMock()
    backend.load_memory.return_value = snapshot()
    backend.save_memory.return_value = MemorySnapshot(A, 2, confirmed_memory(), "token-A")
    service = sessions(backend)
    save = make_save_fn(None, {}, "never-written.json", private_sessions=service, update_panel=True)
    with patch("app.update_memory_from_conversation", return_value=memory("proposal-only-sentinel")):
        status, panel = save(["synthetic conversation"], owner_id=A, request=browser_request(service))
    assert "saved" in status
    assert "Sessions remembered:** 1" in panel
    assert "Preferences stored:** 1" in panel
    assert "confirmed-summary-sentinel" in panel
    assert "proposal-only-sentinel" not in panel
    backend.load_memory.assert_called_once_with("token-A")
    backend.save_memory.assert_called_once()


def test_failed_save_leaves_panel_unchanged():
    backend = MagicMock()
    backend.load_memory.return_value = snapshot()
    backend.save_memory.side_effect = MemoryConflict("Memory changed. Retry saving.")
    service = sessions(backend)
    save = make_save_fn(None, {}, "never-written.json", private_sessions=service, update_panel=True)
    with patch("app.update_memory_from_conversation", return_value=memory("uncommitted-sentinel")):
        status, panel = save(["synthetic conversation"], owner_id=A, request=browser_request(service))
    assert "Retry saving" in status
    assert panel == gr.skip()
    assert "uncommitted-sentinel" not in str((status, panel))


def test_lifecycle_change_during_save_suppresses_status_and_panel():
    backend = MagicMock()
    backend.load_memory.return_value = snapshot()
    scope = {"generation": "before"}

    def completed_save(*args):
        scope["generation"] = "after"
        return MemorySnapshot(A, 2, confirmed_memory(), "token-A")

    backend.save_memory.side_effect = completed_save
    service = sessions(backend)
    save = make_save_fn(None, {}, "never-written.json", private_sessions=service, update_panel=True)
    with patch("app.update_memory_from_conversation", return_value=empty_memory()):
        result = save(["synthetic conversation"], owner_id=A, scope_state=scope,
                      request=browser_request(service))
    assert result == (gr.skip(), gr.skip())


def test_gradio_save_updates_status_and_memory_panel_without_reload():
    backend = MagicMock()
    backend.start_guest.return_value = ("token-A", int(time.time() + 3600))
    backend.load_memory.return_value = snapshot()
    backend.save_memory.return_value = MemorySnapshot(A, 2, confirmed_memory(), "token-A")
    service = sessions(backend)
    demo = build_ui(None, {}, "never-read.json", None, MagicMock(), None, private_sessions=service)
    app = gr.routes.App.create_app(demo, app_kwargs={"routes": service.routes()})

    def turn(**kwargs):
        kwargs["conversation_history"].append("synthetic user turn")
        return "Synthetic answer"

    try:
        with TestClient(app, base_url=ORIGIN) as client, \
                patch("app.run_agent_turn", side_effect=turn), \
                patch("app.update_memory_from_conversation", return_value=empty_memory()):
            headers = {"Origin": ORIGIN}
            assert client.post("/private-session/start", headers=headers,
                               json={"captcha_token": "fixture"}).status_code == 200
            initialized = client.post("/gradio_api/api/refresh_private_session", headers=headers,
                                      json={"data": [[], None, None, None], "session_hash": "save-panel-test"})
            assert initialized.status_code == 200
            submitted = client.post("/gradio_api/api/submit", headers=headers, json={
                "data": ["synthetic question", [], None, None, None, None, None, None],
                "session_hash": "save-panel-test",
            })
            assert submitted.status_code == 200
            saved = client.post("/gradio_api/api/save", headers=headers,
                                json={"data": [None, None, None], "session_hash": "save-panel-test"})
            assert saved.status_code == 200
            status, panel = saved.json()["data"]
            assert "saved" in status
            assert "Sessions remembered:** 1" in panel
            assert "confirmed-summary-sentinel" in panel
    finally:
        demo.close()
