"""Regression tests for public boundaries using synthetic data and no providers."""

import os
from unittest.mock import MagicMock, patch
from uuid import uuid4
from types import SimpleNamespace

import chromadb
import pytest

from access_control import PublicCollection, RuntimePolicy
from agent import build_search_tool
from app import (
    build_ui, load_runtime_memory, make_chat_fn, make_save_fn,
    runtime_memory_status,
)
from config import Config
from gemini_helpers import embed_and_index_chunks
from hybrid_search import _BM25_CACHE, get_bm25_index
from manage_public_corpus import set_public_visibility
from upload_handler import make_upload_fn


PUBLIC = RuntimePolicy()
LOCAL = RuntimePolicy("local_private")


class FixtureEmbeddings:
    """Small deterministic vectors; no model downloads or external embedding API."""

    def __call__(self, input):
        return [[1.0, 0.0, 0.0] for _ in input]

    def embed_query(self, input):
        return self(input)

    def name(self):
        return "privacy_fixture"

    def get_config(self):
        return {}


@pytest.fixture
def collection(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path / "test_chroma"))
    coll = client.create_collection(
        name="privacy_" + uuid4().hex,
        embedding_function=FixtureEmbeddings(),
    )
    coll.add(
        ids=["public", "private", "legacy"],
        documents=["Public showcase passage", "Private sentinel passage",
                   "Unclassified sentinel passage"],
        metadatas=[
            {"visibility": "public", "source_document": "fixture", "page": 0},
            {"visibility": "private", "source_document": "fixture", "page": 0},
            {"source_document": "fixture", "page": 0},
        ],
    )
    yield coll
    client.delete_collection(coll.name)
    _BM25_CACHE.clear()


def test_public_filter_cannot_be_overridden(collection):
    scoped = PublicCollection(collection)
    assert scoped.count() == 1
    assert scoped.get(include=[])["ids"] == ["public"]
    assert scoped.get(where={"visibility": "private"})["ids"] == []
    assert scoped.query(
        query_texts=["sentinel"], n_results=3,
        where={"visibility": "private"},
    )["ids"] == [[]]


@pytest.mark.parametrize("hybrid,reranker", [(False, False), (True, False), (True, True)])
def test_all_search_modes_exclude_private_and_legacy_before_reranking(
        collection, hybrid, reranker):
    s3 = MagicMock()
    ranker = MagicMock()
    ranker.predict.side_effect = lambda pairs: [1.0] * len(pairs)
    with patch("hybrid_search.get_reranker", return_value=ranker):
        search, _ = build_search_tool(
            collection, None, s3, "fixture-bucket",
            use_hybrid=hybrid, use_reranker=reranker, policy=PUBLIC,
        )
        result = search("sentinel")
    assert "Public showcase passage" in result
    assert "Private sentinel" not in result
    assert "Unclassified sentinel" not in result
    # A filename stem alone is insufficient authorization for a public image.
    s3.head_object.assert_not_called()
    if reranker:
        assert ranker.predict.call_args.args[0] == [
            ["sentinel", "Public showcase passage"]]


def test_public_bm25_cannot_reuse_private_cache_and_observes_revocation(collection):
    private_index = get_bm25_index(collection, collection.name)
    assert "private" in private_index["ids"]
    scoped = PublicCollection(collection)
    assert get_bm25_index(scoped, scoped.name)["ids"] == ["public"]
    collection.update(ids=["public"], metadatas=[{"visibility": "private"}])
    index = get_bm25_index(scoped, scoped.name)
    assert index["ids"] == []
    assert index["bm25"] is None


def test_public_citations_use_only_approved_exact_source_key(collection):
    collection.update(ids=["public"], metadatas=[
        {"source_pdf_key": "approved/fixture.pdf"}])
    for hybrid in (False, True):
        with patch("agent.extract_chunk_image", return_value="https://example.invalid/image") as crop:
            search, _ = build_search_tool(
                collection, None, MagicMock(), "fixture-bucket",
                use_hybrid=hybrid, policy=PUBLIC,
            )
            search("showcase")
            assert crop.call_args.kwargs["source_pdf_key"] == "approved/fixture.pdf"


def test_new_ingestion_cannot_grant_public_access_from_chunk_payload(collection):
    embed_and_index_chunks([
        {"chunk_id": "new", "text": "new content", "visibility": "public"}
    ], collection)
    assert collection.get(ids=["new"])["metadatas"][0]["visibility"] == "private"
    assert "new" not in PublicCollection(collection).get(include=[])["ids"]


def test_public_startup_does_not_read_legacy_memory():
    with patch("app.load_memory") as load:
        assert load_runtime_memory("private-memory.json", PUBLIC) == {}
        load.assert_not_called()
        load_runtime_memory("private-memory.json", LOCAL)
        load.assert_called_once_with("private-memory.json")


def test_public_chat_does_not_inject_shared_memory_or_other_session_history():
    legacy = {"facts": ["legacy-private-sentinel"], "preferences": {},
              "session_summaries": ["legacy-summary-sentinel"]}
    session_a, session_b = [], []
    received = []

    def turn(**kwargs):
        assert "legacy-private-sentinel" not in kwargs["generation_config"].system_instruction
        received.append(list(kwargs["conversation_history"]))
        kwargs["conversation_history"].append(kwargs["user_message"])
        return "Public answer"

    chat = make_chat_fn(None, legacy, "memory.json", None, MagicMock(), None, policy=PUBLIC)
    with patch("app.run_agent_turn", side_effect=turn):
        result_a = chat("Session A question", [], session_a, "Standard Vector Search", False, False)
        result_b = chat("Session B question", [], session_b, "Standard Vector Search", False, False)
    assert received == [[], []]
    assert session_a == session_b == []
    assert result_a[1] == ["Session A question"]
    assert result_b[1] == ["Session B question"]
    assert "legacy-summary-sentinel" not in result_a[3]
    assert "legacy-summary-sentinel" not in runtime_memory_status(legacy, PUBLIC)


def test_failed_turn_keeps_original_session_state():
    history = ["previous completed turn"]

    def fail(**kwargs):
        kwargs["conversation_history"].extend(["user", "tool call", "tool response"])
        raise RuntimeError("private-provider-sentinel")

    chat = make_chat_fn(None, {}, "unused.json", None, MagicMock(), None, policy=PUBLIC)
    with patch("app.run_agent_turn", side_effect=fail):
        result = chat("question", [], history, "Standard Vector Search", False, False)
    assert history == ["previous completed turn"]
    assert result[1] == history
    assert "private-provider-sentinel" not in str(result)


def test_direct_public_mutations_fail_before_any_side_effect():
    provider, s3, coll = MagicMock(), MagicMock(), MagicMock()
    save = make_save_fn(provider, {}, "never-written.json", policy=PUBLIC)
    upload = make_upload_fn(s3, "fixture-bucket", coll, policy=PUBLIC)
    with patch("app.save_memory") as write, patch("app.update_memory_from_conversation") as summarize:
        with pytest.raises(PermissionError):
            save(["private conversation"])
        with pytest.raises(PermissionError):
            list(upload(["/nonexistent/file.pdf"]))
        write.assert_not_called()
        summarize.assert_not_called()
    assert not s3.mock_calls
    assert not coll.mock_calls
    assert not provider.mock_calls


def test_local_workflow_remains_available():
    provider = MagicMock()
    memory = {"facts": [], "preferences": {}, "session_summaries": []}
    with patch("app.update_memory_from_conversation", return_value=memory) as summarize, patch("app.save_memory") as write:
        save = make_save_fn(provider, memory, "fixture-memory.json", policy=LOCAL)
        assert save(["fixture conversation"]) == "✅ Memory saved!"
        summarize.assert_called_once()
        write.assert_called_once_with(memory, "fixture-memory.json")
    assert list(make_upload_fn(None, None, None, policy=LOCAL)([])) == ["⚠️ No files selected."]


def test_invalid_and_hosted_local_modes_fail_closed():
    with pytest.raises(ValueError):
        RuntimePolicy("typo")
    with pytest.raises(ValueError):
        LOCAL.validate_launch(share=True)
    with patch.dict(os.environ, {"APP_MODE": "local_private", "SPACE_ID": "fixture/space"}):
        with pytest.raises(ValueError):
            Config()


def test_public_ui_never_serializes_legacy_memory():
    legacy = {"facts": ["ui-private-sentinel"], "session_summaries": ["ui-private-sentinel"]}
    demo = build_ui(None, legacy, "unused.json", None, MagicMock(), None, policy=PUBLIC)
    assert "ui-private-sentinel" not in str(demo.config)
    files = [c for c in demo.config["components"] if c["type"] == "file"]
    assert all(c["props"]["visible"] is False for c in files)
    demo.close()


def test_explicit_publication_and_revocation_do_not_publish_other_chunks(collection):
    set_public_visibility(
        collection, ["legacy"], publish=True,
        source_pdf_key="approved/fixture.pdf",
    )
    scoped = PublicCollection(collection)
    assert set(scoped.get(include=[])["ids"]) == {"public", "legacy"}
    assert collection.get(ids=["legacy"])["metadatas"][0]["source_pdf_key"] == "approved/fixture.pdf"
    assert "private" not in scoped.get(include=[])["ids"]
    set_public_visibility(collection, ["legacy"], publish=False)
    assert scoped.get(include=[])["ids"] == ["public"]


def test_invalid_publication_is_atomic_before_update(collection):
    with patch.object(type(collection), "update") as update:
        with pytest.raises(ValueError):
            set_public_visibility(collection, ["private", "missing"], publish=True,
                                  source_pdf_key="approved/fixture.pdf")
        with pytest.raises(ValueError):
            set_public_visibility(collection, ["private"], publish=True)
        with pytest.raises(ValueError):
            set_public_visibility(collection, ["private"], publish=True,
                                  source_pdf_key="../private.pdf")
        update.assert_not_called()


def test_citation_cache_is_scoped_to_exact_source_not_filename():
    from visual_grounding_helper import extract_chunk_image

    s3 = MagicMock()
    s3.generate_presigned_url.return_value = "https://example.invalid/citation"
    keys = []
    for source in ("approved/report.pdf", "private/report.pdf"):
        with patch("visual_grounding_helper.DYNAMIC_CROPPING_ENABLED", True):
            extract_chunk_image(s3, "fixture-bucket", source,
                                [0, 0, 1, 1], 0, "same-chunk", "report")
        keys.append(s3.head_object.call_args.kwargs["Key"])
    assert keys[0] != keys[1]
    assert all(key.startswith("output/chunk_images/") for key in keys)
    s3.get_object.assert_not_called()


@pytest.mark.parametrize("policy,host", [(PUBLIC, "0.0.0.0"), (LOCAL, "127.0.0.1")])
def test_launch_enforces_mode_and_file_boundary(policy, host):
    from app import main

    settings = SimpleNamespace(ACCESS_POLICY=policy, GEMINI_API_KEYS=["mock"],
                               S3_BUCKET_NAME="fixture-bucket")
    demo = MagicMock()
    with patch("app.settings", settings), patch("sys.argv", ["app.py"]), \
            patch("app.create_gemini_router"), patch("app.create_s3_client"), \
            patch("app.load_chroma_collection"), patch("app.load_memory", return_value={}) as load, \
            patch("app.build_ui", return_value=demo):
        main()
    options = demo.launch.call_args.kwargs
    assert options["server_name"] == host
    assert options["strict_cors"] is True
    assert options["show_error"] is (not policy.is_public)
    assert options["max_file_size"] == (0 if policy.is_public else "20mb")
    if policy.is_public:
        load.assert_not_called()
        assert any(path.endswith("memory.json") for path in options["blocked_paths"])
        assert any(path.endswith("chroma_db") for path in options["blocked_paths"])
    else:
        load.assert_called_once()


def test_local_share_rejected_before_initializing_any_provider():
    from app import main

    with patch("app.settings", SimpleNamespace(ACCESS_POLICY=LOCAL)), \
            patch("sys.argv", ["app.py", "--share"]), \
            patch("app.create_gemini_router") as provider:
        with pytest.raises(ValueError):
            main()
        provider.assert_not_called()


def test_public_framework_upload_route_rejects_file_body(tmp_path):
    import gradio as gr
    from fastapi.testclient import TestClient

    demo = build_ui(None, {}, "unused.json", None, MagicMock(), None, policy=PUBLIC)
    demo.max_file_size = 0
    app = gr.routes.App.create_app(demo)
    app.uploaded_file_dir = str(tmp_path / "uploads")
    try:
        with TestClient(app) as client:
            response = client.post(
                "/gradio_api/upload",
                files={"files": ("fixture.pdf", b"%PDF-fixture", "application/pdf")},
            )
        assert response.status_code == 413
        assert not (tmp_path / "uploads").exists()
    finally:
        demo.close()
