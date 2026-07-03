"""Unit tests for the API-provider layer (backend: api). No real API calls."""

from __future__ import annotations

import types
from typing import Any

import pytest

from modelrack.providers import get_provider, register_provider
from modelrack.providers.anthropic_provider import AnthropicProvider
from modelrack.providers.base import Provider, ProviderError
from modelrack.schemas.resolved_model import ResolvedModel


def _resolved(api_model_id: str = "claude-opus-4-8", **merged: Any) -> ResolvedModel:
    return ResolvedModel(
        model_id="claude",
        display_name="Claude",
        type="language",
        backend="api",
        provider="anthropic",
        api_model_id=api_model_id,
        merged_config=merged,
        param_schema={},
        load_hints={},
        hardware={},
        environment={},
        serving={},
    )


# ===========================================================================
# _build_request
# ===========================================================================

def test_build_request_core_fields():
    req = AnthropicProvider._build_request(
        _resolved(), {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 256}
    )
    assert req["model"] == "claude-opus-4-8"
    assert req["max_tokens"] == 256
    assert req["messages"] == [{"role": "user", "content": "hi"}]
    assert "system" not in req


def test_build_request_system_and_default_max_tokens():
    req = AnthropicProvider._build_request(
        _resolved(), {"messages": [{"role": "user", "content": "hi"}], "system": "be terse"}
    )
    assert req["system"] == "be terse"
    assert req["max_tokens"] == 1024  # default


def test_build_request_provider_params_passthrough_core_wins():
    req = AnthropicProvider._build_request(
        _resolved(),
        {
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 512,
            "provider_params": {"thinking": {"type": "adaptive"}, "max_tokens": 999},
        },
    )
    # native field flows through
    assert req["thinking"] == {"type": "adaptive"}
    # normalized core field wins over the same key in provider_params
    assert req["max_tokens"] == 512


def test_build_request_missing_messages_raises():
    with pytest.raises(ProviderError, match="messages is required"):
        AnthropicProvider._build_request(_resolved(), {"max_tokens": 10})


# ===========================================================================
# _map_response
# ===========================================================================

def test_map_response_concatenates_text_and_reads_usage():
    resp = types.SimpleNamespace(
        content=[
            types.SimpleNamespace(type="thinking", thinking="..."),
            types.SimpleNamespace(type="text", text="Hello "),
            types.SimpleNamespace(type="text", text="world"),
        ],
        model="claude-opus-4-8",
        stop_reason="end_turn",
        usage=types.SimpleNamespace(input_tokens=12, output_tokens=3),
    )
    out = AnthropicProvider._map_response(resp)
    assert out == {
        "text": "Hello world",
        "model": "claude-opus-4-8",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 12, "output_tokens": 3},
    }


# ===========================================================================
# infer end-to-end (fake client)
# ===========================================================================

def test_infer_calls_sdk_and_maps(monkeypatch):
    captured = {}

    class _Messages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(
                content=[types.SimpleNamespace(type="text", text="pong")],
                model=kwargs["model"],
                stop_reason="end_turn",
                usage=types.SimpleNamespace(input_tokens=5, output_tokens=1),
            )

    class _FakeClient:
        messages = _Messages()

    prov = AnthropicProvider()
    monkeypatch.setattr(prov, "_client", lambda resolved: _FakeClient())

    data = prov.infer(_resolved(), {"messages": [{"role": "user", "content": "ping"}]})
    assert data["text"] == "pong"
    assert captured["model"] == "claude-opus-4-8"
    assert captured["messages"] == [{"role": "user", "content": "ping"}]


def test_infer_wraps_sdk_error(monkeypatch):
    class _Boom:
        def create(self, **kwargs):
            raise RuntimeError("429 overloaded")

    prov = AnthropicProvider()
    monkeypatch.setattr(prov, "_client", lambda resolved: types.SimpleNamespace(messages=_Boom()))
    with pytest.raises(ProviderError, match="Anthropic request failed"):
        prov.infer(_resolved(), {"messages": [{"role": "user", "content": "x"}]})


# ===========================================================================
# credentials + registry
# ===========================================================================

def test_resolve_api_key_reads_named_env(monkeypatch):
    monkeypatch.setenv("MY_KEY", "secret-123")
    r = _resolved(api_key_env="MY_KEY")
    assert Provider.resolve_api_key(r, "ANTHROPIC_API_KEY") == "secret-123"


def test_resolve_api_key_none_when_unset(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert Provider.resolve_api_key(_resolved(), "ANTHROPIC_API_KEY") is None


def test_get_provider_known_and_cached():
    a = get_provider("anthropic")
    b = get_provider("Anthropic")  # case-insensitive, cached
    assert isinstance(a, AnthropicProvider)
    assert a is b


def test_get_provider_unknown_and_missing():
    with pytest.raises(ProviderError, match="Unknown API provider"):
        get_provider("nope")
    with pytest.raises(ProviderError, match="no 'provider' set"):
        get_provider(None)


def test_register_provider_roundtrip():
    class _Fake(Provider):
        name = "fake"

        def infer(self, resolved, payload):
            return {"ok": True}

    register_provider("fake-test", _Fake)
    assert isinstance(get_provider("fake-test"), _Fake)


# ===========================================================================
# ModelRack routing: backend: api -> provider (not the subprocess client)
# ===========================================================================

def test_modelrack_routes_api_model_to_provider(tmp_path, monkeypatch):
    import modelrack.providers as providers_pkg
    from modelrack import ModelRack

    (tmp_path / "registry.yaml").write_text(
        "version: '1.0'\n"
        "models:\n"
        "  claude:\n"
        "    type: language\n"
        "    backend: api\n"
        "    provider: anthropic\n"
        "    api_model_id: claude-opus-4-8\n"
        "    added_at: '2026-07-03T00:00:00'\n",
        encoding="utf-8",
    )

    captured = {}

    class _FakeProvider:
        def infer(self, resolved, payload):
            captured["api_model_id"] = resolved.api_model_id
            captured["payload"] = payload
            return {"text": "routed"}

    monkeypatch.setattr(providers_pkg, "get_provider", lambda name: _FakeProvider())

    hub = ModelRack(models_dir=tmp_path)
    result = hub.infer("claude", {"messages": [{"role": "user", "content": "hi"}]})

    assert result == {"success": True, "data": {"text": "routed"}, "error": None}
    assert captured["api_model_id"] == "claude-opus-4-8"
    assert captured["payload"]["messages"] == [{"role": "user", "content": "hi"}]
