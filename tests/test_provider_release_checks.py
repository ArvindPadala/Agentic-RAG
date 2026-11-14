"""Release-probe measurements must not masquerade as generation or load passes."""

import json
from unittest.mock import MagicMock
from unittest.mock import patch

from google.genai.errors import APIError
import httpx
import pytest

from eval.check_provider_release import SyntheticTransport, load_request, percentile, probe_models, summarize_load


def test_model_probe_only_gets_metadata_with_single_attempt_and_no_generation():
    client = MagicMock()
    client.models.get.return_value = MagicMock(supported_actions=["generateContent"])
    rows = []
    probe_models([client], {"models/fixture": ["fallback"]}, 10000, rows.append)
    assert rows[0]["status"] == "metadata_available"
    assert rows[0]["credential_slot"] == 1
    config = client.models.get.call_args.kwargs["config"]
    assert config.http_options.timeout == 10000
    assert config.http_options.retry_options.attempts == 1
    client.models.generate_content.assert_not_called()


def test_model_auth_failure_skips_remaining_models_for_that_credential_without_leaking_body():
    client = MagicMock()
    client.models.get.side_effect = APIError(403, {"message": "private provider error sentinel"})
    rows = []
    probe_models([client], {"models/a": ["primary"], "models/b": ["fallback"]}, 10000, rows.append)
    assert [row["status"] for row in rows] == ["failed", "skipped"]
    assert rows[0]["error"]["code"] == 403
    assert "private provider" not in json.dumps(rows)
    assert client.models.get.call_count == 1


def request(request_id="loadcase-000", *, tool=False):
    contents = [{"role": "user", "parts": [{"text": f"Request {request_id}"}]}]
    if tool:
        contents.append({"role": "user", "parts": [{"functionResponse": {
            "name": "search_knowledge_base", "response": {"result": "Evidence [1] public passage"}}}]})
    return httpx.Request("POST", "https://example.invalid/models/fixture:generateContent",
                         json={"contents": contents})


def test_synthetic_transport_injects_per_request_failures_then_returns_tools_and_citations():
    transport = SyntheticTransport(0, 1)
    assert transport.handle(request()).status_code == 429
    planned = transport.handle(request()).json()
    assert planned["candidates"][0]["content"]["parts"][0]["functionCall"]["name"] == "search_knowledge_base"
    answered = transport.handle(request(tool=True)).json()
    assert "loadcase-000 [1]" in answered["candidates"][0]["content"]["parts"][0]["text"]
    assert transport.handle(request("loadcase-001")).status_code == 200
    assert transport.active == 0
    assert sum(event["status"] == 429 for event in transport.events) == 1


def test_transport_rejects_mixed_request_markers():
    transport = SyntheticTransport(0, 0)
    with pytest.raises(ValueError, match="missing or mixed"):
        transport.handle(request("loadcase-000 plus loadcase-001"))
    assert transport.events == []


def test_latency_percentiles_include_failed_requests_and_do_not_round_into_a_pass():
    rows = [{"status": "completed", "service_seconds": 1, "queue_seconds": 0, "total_seconds": 1},
            {"status": "failed", "service_seconds": 3, "queue_seconds": 2, "total_seconds": 5}]
    result = summarize_load(rows, 5)
    assert result["completed"] == 1 and result["failed"] == 1
    assert result["service_p95_seconds"] == 3
    assert result["total_p95_seconds"] == 5
    assert result["completed_requests_per_second"] == 0.2
    assert percentile([], 0.95) is None


def test_load_worker_runs_actual_sdk_graph_and_validates_its_own_evidence():
    import time
    from google import genai
    from llm_router import GeminiRouter, RouterPolicy
    from tests.test_runtime_privacy import ready_collection

    result = {"chunk_id": "public-fixture", "text": "Public climate evidence", "source_document": "Public report",
              "source_pdf_key": "approved/public.pdf", "page": 0, "bbox": [0.1, 0.1, 0.9, 0.9],
              "score": 1.0, "chunk_type": "text"}
    collection = ready_collection()
    collection.get.return_value = {"ids": [result["chunk_id"]], "metadatas": [{
        "source_pdf_key": result["source_pdf_key"], "page": 0}]}
    synthetic, original = SyntheticTransport(0, 0), genai.Client
    with httpx.Client(transport=httpx.MockTransport(synthetic.handle)) as http:
        def client(**kwargs):
            kwargs["http_options"] = kwargs["http_options"].model_copy(update={"httpx_client": http})
            return original(**kwargs)

        with patch("llm_router.genai.Client", side_effect=client):
            router = GeminiRouter(["synthetic-key"], models=["models/synthetic-primary"], policy=RouterPolicy())
        try:
            with patch("agent.search_chroma", return_value=[result]):
                row = load_request("loadcase-000", time.perf_counter(), router, collection, "vector")
        finally:
            router.clients[0].close()
    assert row["status"] == "completed"
    assert row["context_isolated"] is True
    assert row["execution"]["provider_attempts"] == len(synthetic.events) == 3
    assert row["cited_ids"] == [1]
    assert row["queue_seconds"] >= 0
    assert row["total_seconds"] >= row["service_seconds"]
