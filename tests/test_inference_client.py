"""Tests for InferenceClient (routing, auto-start, error handling) with mocked httpx."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import modelrack.inference_client as ic_mod
from modelrack.exceptions import InferenceError, ServerNotRunningError
from modelrack.inference_client import InferenceClient


class FakePM:
    """Stand-in for ProcessManager exposing only what InferenceClient uses."""

    def __init__(self, url: str | None = None) -> None:
        self._url = url
        self.started = False

    def get_server_url(self, _model_id: str) -> str | None:
        return self._url

    def start(self, _model_id: str) -> SimpleNamespace:
        self.started = True
        self._url = "http://127.0.0.1:7801"
        return SimpleNamespace(url=self._url)


class FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self) -> dict:
        return self._payload


def test_infer_routes_to_correct_url(monkeypatch: pytest.MonkeyPatch):
    captured: dict = {}

    def fake_post(url, json, timeout):  # noqa: A002
        captured["url"] = url
        captured["json"] = json
        return FakeResponse(200, {"success": True, "data": {"ok": 1}})

    monkeypatch.setattr(ic_mod.httpx, "post", fake_post)
    client = InferenceClient(FakePM(url="http://127.0.0.1:7801"))
    result = client.infer("demo", {"prompt": "hi"})

    assert captured["url"] == "http://127.0.0.1:7801/infer"
    assert captured["json"] == {"payload": {"prompt": "hi"}}
    assert result["data"]["ok"] == 1


def test_infer_auto_start(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        ic_mod.httpx, "post", lambda *_a, **_k: FakeResponse(200, {"success": True})
    )
    pm = FakePM(url=None)
    client = InferenceClient(pm)
    client.infer("demo", {"prompt": "hi"}, auto_start=True)
    assert pm.started is True


def test_infer_no_auto_start_raises():
    client = InferenceClient(FakePM(url=None))
    with pytest.raises(ServerNotRunningError):
        client.infer("demo", {"prompt": "hi"}, auto_start=False)


def test_infer_non_200_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ic_mod.httpx, "post", lambda *_a, **_k: FakeResponse(500, text="boom"))
    client = InferenceClient(FakePM(url="http://127.0.0.1:7801"))
    with pytest.raises(InferenceError) as exc:
        client.infer("demo", {"prompt": "hi"})
    assert "boom" in str(exc.value)


def test_unload_and_info(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        ic_mod.httpx, "post", lambda *_a, **_k: FakeResponse(200, {"success": True})
    )
    monkeypatch.setattr(
        ic_mod.httpx, "get", lambda *_a, **_k: FakeResponse(200, {"success": True, "data": {}})
    )
    client = InferenceClient(FakePM(url="http://127.0.0.1:7801"))
    assert client.unload("demo")["success"] is True
    assert client.info("demo")["success"] is True
