"""Anthropic (Claude) provider — Messages API.

Normalized payload (for `language` / `vision_language` models):
    messages          (list, required)   standard chat messages
    max_tokens        (int)   default 1024
    system            (str)   optional system prompt
    provider_params   (dict)  native escape hatch — passed straight to messages.create
                              (e.g. thinking, tools, tool_choice, temperature)

Normalized response: {"text", "model", "stop_reason", "usage": {input_tokens, output_tokens}}.

The `anthropic` SDK is imported lazily; install with `pip install 'modelrack[anthropic]'`.
"""

from __future__ import annotations

from typing import Any

from modelrack.providers.base import Provider, ProviderError
from modelrack.schemas.resolved_model import ResolvedModel

_DEFAULT_MODEL = "claude-opus-4-8"
_DEFAULT_KEY_ENV = "ANTHROPIC_API_KEY"


class AnthropicProvider(Provider):
    """Serve Claude models via the Anthropic Messages API."""

    name = "anthropic"

    def infer(self, resolved: ResolvedModel, payload: dict[str, Any]) -> dict[str, Any]:
        client = self._client(resolved)
        request = self._build_request(resolved, payload)
        try:
            resp = client.messages.create(**request)
        except Exception as exc:  # noqa: BLE001 - normalize any SDK/API error
            raise ProviderError(f"Anthropic request failed: {exc}") from exc
        return self._map_response(resp)

    # --- seam (patched in tests) ---------------------------------------------
    def _client(self, resolved: ResolvedModel) -> Any:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise ProviderError(
                "The 'anthropic' package is required for provider 'anthropic'. "
                "Install it with:  pip install 'modelrack[anthropic]'"
            ) from exc
        key = self.resolve_api_key(resolved, _DEFAULT_KEY_ENV)
        # No key set → let the SDK resolve credentials from its own chain
        # (ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN / an `ant auth login` profile).
        return anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()

    # --- pure helpers (unit-tested) ------------------------------------------
    @staticmethod
    def _build_request(resolved: ResolvedModel, payload: dict[str, Any]) -> dict[str, Any]:
        messages = payload.get("messages")
        if not messages:
            raise ProviderError("payload.messages is required for this Anthropic model")

        core: dict[str, Any] = {
            "model": resolved.api_model_id or _DEFAULT_MODEL,
            "max_tokens": int(payload.get("max_tokens", 1024)),
            "messages": messages,
        }
        if payload.get("system"):
            core["system"] = payload["system"]

        # provider_params supplies native fields not covered by the normalized surface
        # (thinking, tools, temperature, …). Core normalized fields always win on overlap.
        extra = payload.get("provider_params")
        if isinstance(extra, dict):
            return {**extra, **core}
        return core

    @staticmethod
    def _map_response(resp: Any) -> dict[str, Any]:
        text = ""
        for block in getattr(resp, "content", None) or []:
            if getattr(block, "type", None) == "text":
                text += getattr(block, "text", "") or ""
        usage = getattr(resp, "usage", None)
        return {
            "text": text.strip(),
            "model": getattr(resp, "model", None),
            "stop_reason": getattr(resp, "stop_reason", None),
            "usage": {
                "input_tokens": getattr(usage, "input_tokens", None) if usage else None,
                "output_tokens": getattr(usage, "output_tokens", None) if usage else None,
            },
        }
