"""Smoke tests for the hub REST API using FastAPI's TestClient."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from modelrack.api.server import create_app


def _client(models_dir: Path) -> TestClient:
    return TestClient(create_app(models_dir))


def test_health(models_dir: Path):
    with _client(models_dir) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["status"] == "ok"


def test_list_models(models_dir: Path):
    with _client(models_dir) as client:
        body = client.get("/models").json()
        ids = {m["model_id"] for m in body["data"]}
        assert {"demo-image", "demo-llm"} <= ids


def test_list_models_filtered(models_dir: Path):
    with _client(models_dir) as client:
        body = client.get("/models", params={"type": "language"}).json()
        assert [m["model_id"] for m in body["data"]] == ["demo-llm"]


def test_get_schema(models_dir: Path):
    with _client(models_dir) as client:
        body = client.get("/models/demo-image/schema").json()
        assert "num_inference_steps" in body["data"]


def test_resolve_with_runtime_params(models_dir: Path):
    with _client(models_dir) as client:
        resp = client.post(
            "/models/demo-image/resolve",
            json={"runtime_params": {"num_inference_steps": 20}},
        )
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["merged_config"]["defaults"]["num_inference_steps"] == 20


def test_resolve_out_of_range_fails(models_dir: Path):
    with _client(models_dir) as client:
        resp = client.post(
            "/models/demo-image/resolve",
            json={"runtime_params": {"num_inference_steps": 9999}},
        )
        body = resp.json()
        assert body["success"] is False
        assert body["error"]


def test_unknown_model_404(models_dir: Path):
    with _client(models_dir) as client:
        resp = client.get("/models/nope")
        assert resp.status_code == 404
        assert resp.json()["success"] is False
