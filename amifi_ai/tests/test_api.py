# amifi_ai/tests/test_api.py
"""FastAPI endpoint tests using TestClient (no live server needed)."""

import io

from fastapi.testclient import TestClient

from amifi_ai.api.main import app

client = TestClient(app, raise_server_exceptions=False)


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["service"] == "FinEdge AI"


def test_health_ok():
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] in ("ok", "degraded")
    assert "checks" in data
    assert data["checks"]["model_loaded"] is True
    assert data["checks"]["tokenizer_loaded"] is True


def test_classify_regex_path():
    r = client.post("/api/v1/classify", json={
        "text": "HDFC Bank Rs 2500 debited at AMAZON on 05-03-24",
        "use_llm": False
    })
    assert r.status_code == 200
    data = r.json()
    assert data["merchant"] == "Amazon"
    assert data["type"] == "debit"
    assert data["confidence"] >= 0.9
    assert data["tokens_generated"] == 0
    assert "x-request-id" in r.headers


def test_classify_empty_text_422():
    r = client.post("/api/v1/classify", json={"text": ""})
    assert r.status_code == 422


def test_classify_text_too_long_422():
    r = client.post("/api/v1/classify", json={"text": "x" * 501})
    assert r.status_code == 422


def test_parse_csv_success():
    with open("tests/fixtures/sample_bank.csv", "rb") as f:
        r = client.post("/api/v1/parse-csv", files={"file": ("sample_bank.csv", f, "text/csv")})
    assert r.status_code == 200
    data = r.json()
    assert data["total_rows"] == 7
    assert len(data["transactions"]) == 7


def test_parse_csv_wrong_type_415():
    fake_txt = io.BytesIO(b"not a csv file")
    r = client.post("/api/v1/parse-csv", files={"file": ("test.txt", fake_txt, "text/plain")})
    assert r.status_code == 415


def test_parse_csv_empty_file_422():
    empty = io.BytesIO(b"")
    r = client.post("/api/v1/parse-csv", files={"file": ("empty.csv", empty, "text/csv")})
    assert r.status_code == 422


def test_request_id_header_present():
    r = client.get("/")
    assert "x-request-id" in r.headers


def test_response_time_header_present():
    r = client.get("/api/v1/health")
    assert "x-response-time-ms" in r.headers
