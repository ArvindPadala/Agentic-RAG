"""Offline public-corpus integrity and citation-boundary regression tests."""

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.parse import unquote

import chromadb
import fitz
import gradio as gr
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from access_control import PublicCollection
from agent import build_search_tool
from app import build_ui, clean_response, extract_image_urls, make_chat_fn
from gemini_helpers import embed_and_index_chunks
from showcase_corpus import ShowcaseCorpus, download_pdf, extract_chunks, prepare_showcase
from tests.test_runtime_privacy import FixtureEmbeddings, ready_collection


@pytest.fixture
def document(tmp_path):
    source = {"id": "fixture", "title": "Fixture climate report (2023)",
              "url": "https://www.ipcc.ch/fixture.pdf"}
    path = tmp_path / "pdfs" / "fixture.pdf"
    path.parent.mkdir()
    with fitz.open() as pdf:
        page = pdf.new_page()
        page.insert_text((72, 72), "Solar energy costs fell in the assessed period with high confidence.")
        page = pdf.new_page()
        page.insert_text((72, 72), "Mitigation pathways reduce emissions and future warming risks.")
        pdf.save(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    source["sha256"] = digest
    return source, path, digest


def test_checksum_rejects_changed_cached_pdf(tmp_path, document):
    source, path, _ = document
    assert download_pdf(source, tmp_path)[0] == path
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum"):
        download_pdf(source, tmp_path)


def test_real_pdf_coordinates_and_file_serving_boundary(tmp_path, document):
    source, path, digest = document
    chunks = extract_chunks(path, source, digest)
    assert [chunk["page"] for chunk in chunks] == [0, 1]
    corpus = ShowcaseCorpus(tmp_path, [document])
    url = corpus.render_citation(chunks[1])
    image = Path(unquote(url.split("=", 1)[1]))
    assert image.read_bytes().startswith(b"\x89PNG")
    assert corpus.render_citation({**chunks[0], "source_pdf_key": "../../private.pdf"}) is None
    assert corpus.render_citation({**chunks[0], "bbox": [0, 0, float("nan"), 1]}) is None
    assert corpus.render_citation({**chunks[0], "page": 99}) is None
    assert extract_image_urls(f"[Evidence]({url})") == [url]
    assert url not in clean_response(f"[Evidence]({url})")
    demo = build_ui(None, {}, "unused.json", None, ready_collection(), None, showcase=corpus)
    app = gr.mount_gradio_app(FastAPI(), demo, path="", allowed_paths=[str(corpus.images)])
    with TestClient(app) as client:
        assert client.get(url).status_code == 200
        assert client.get("/gradio_api/file=" + str(path)).status_code == 403
        config = client.get("/config").json()
        assert any("IPCC AR6" in str(component.get("props", {}).get("value", ""))
                   for component in config["components"])
    chat = make_chat_fn(None, {}, "unused.json", None, ready_collection(), None, showcase=corpus)
    with patch("app.run_agent_turn", return_value=f"Public answer. [Evidence]({url})"):
        result = chat("question", [], [], "Standard Vector Search", False, False)
    assert url in result[2]
    assert url not in result[0][-1]["content"]


def test_provisioning_is_complete_idempotent_and_isolated(tmp_path, document):
    source, _, _ = document
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))

    def init(path, name):
        return client.get_or_create_collection(name, embedding_function=FixtureEmbeddings())

    with patch("showcase_corpus.curated_sources", return_value=[source]), patch(
            "gemini_helpers.init_chroma_collection", side_effect=init):
        collection, corpus = prepare_showcase(tmp_path)
        again, _ = prepare_showcase(tmp_path)
    assert collection.name == again.name
    assert PublicCollection(collection).count() == collection.count() == 2
    for hybrid in (False, True):
        search, _ = build_search_tool(collection, None, None, None, use_hybrid=hybrid,
                                      citation_renderer=corpus.render_citation)
        response = search("solar costs")
        assert "Fixture climate report (2023)" in response
        assert "**PDF Page:** 1" in response
        assert "https://www.ipcc.ch/fixture.pdf#page=1" in response
        assert extract_image_urls(response)


def test_incomplete_index_is_never_published(tmp_path, document):
    source, _, _ = document
    collection = MagicMock()
    collection.get.return_value = {"ids": []}
    with patch("showcase_corpus.curated_sources", return_value=[source]), patch(
            "gemini_helpers.init_chroma_collection", return_value=collection), patch(
            "gemini_helpers.embed_and_index_chunks"), patch(
            "showcase_corpus.set_public_visibility") as publish:
        with pytest.raises(ValueError, match="complete chunk manifest"):
            prepare_showcase(tmp_path)
    publish.assert_not_called()


def test_empty_public_search_does_not_offer_uploads():
    collection = MagicMock()
    collection.get.return_value = {"ids": []}
    search, _ = build_search_tool(collection, None, None, None)
    response = search("question")
    assert "Public uploads are unavailable" in response
    assert "administrator" in response


def test_empty_ingestion_skips_invalid_empty_id_lookup():
    collection = MagicMock()
    assert embed_and_index_chunks([], collection) == 0
    assert embed_and_index_chunks([{"text": "  "}], collection) == 0
    collection.get.assert_not_called()
    collection.add.assert_not_called()
