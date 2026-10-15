import json

import pytest
from fastapi.testclient import TestClient

from query_tool.api.main import create_app
from tests.conftest import FakeLLMProvider


@pytest.fixture
def client(make_service):
    with TestClient(create_app(make_service())) as c:
        yield c


def test_health(client):
    body = client.get("/api/v1/health").json()
    assert body["llm_configured"] is True
    assert body["documents"] == 1
    assert body["chunks"] >= 1


def test_query(client):
    response = client.post("/api/v1/query", json={"question": "Refund policy?"})
    assert response.status_code == 200
    assert response.headers["X-Request-ID"]
    assert response.json()["approved"] is True


def test_query_validation(client):
    assert client.post("/api/v1/query", json={"question": "a"}).status_code == 422
    assert client.post("/api/v1/query", json={"question": "abc", "top_k": 99}).status_code == 422


def test_query_stream_ndjson(client):
    with client.stream("POST", "/api/v1/query/stream", json={"question": "Refund?"}) as r:
        assert r.headers["content-type"].startswith("application/x-ndjson")
        events = [json.loads(line) for line in r.iter_lines() if line]
    assert events[0]["event"] == "start"
    assert events[-1]["event"] == "result"


def test_document_lifecycle(client):
    files = [
        ("files", ("notes.txt", b"The office closes at 6pm on Fridays.", "text/plain")),
        ("files", ("image.png", b"\x89PNG", "image/png")),
    ]
    body = client.post("/api/v1/documents", files=files).json()
    assert [d["source"] for d in body["indexed"]] == ["notes.txt"]
    assert "unsupported file type" in body["skipped"][0]

    docs = client.get("/api/v1/documents").json()
    assert {d["source"] for d in docs} == {"billing.md", "notes.txt"}

    doc_id = body["indexed"][0]["doc_id"]
    assert client.delete(f"/api/v1/documents/{doc_id}").status_code == 204
    assert client.delete(f"/api/v1/documents/{doc_id}").status_code == 404


def test_unconfigured_llm_returns_503(make_service):
    service = make_service(FakeLLMProvider(configured=False))
    with TestClient(create_app(service)) as c:
        response = c.post("/api/v1/query/stream", json={"question": "Refund policy?"})
    assert response.status_code == 503
    assert "GROQ_API_KEY" in response.json()["detail"]
