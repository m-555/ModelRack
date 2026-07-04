"""Google (Gemini) provider — Generative Language API via the `google-genai` SDK.

Normalized payload (for `language` / `vision_language` models):
    messages          (list, required)   standard chat messages ({role, content})
    max_tokens        (int)   default 1024  -> max_output_tokens
    system            (str)   optional system prompt -> system_instruction
    provider_params   (dict)  native escape hatch merged into the request `config`
                              (e.g. temperature, top_p, thinking_config, tools,
                              response_mime_type, safety_settings)

Normalized response: {"text", "model", "stop_reason", "usage": {input_tokens, output_tokens}}.

The `google-genai` SDK is imported lazily; install with `pip install 'modelrack[google]'`.
"""

from __future__ import annotations

import os
from typing import Any

from modelrack.providers.base import Provider, ProviderError
from modelrack.schemas.resolved_model import ResolvedModel

_DEFAULT_MODEL = "gemini-2.5-flash"
_DEFAULT_KEY_ENV = "GEMINI_API_KEY"
_TRUTHY = {"1", "true", "yes", "on"}

# modelrack uses OpenAI/Anthropic-style roles; Gemini uses "user"/"model".
_ROLE_MAP = {"user": "user", "assistant": "model", "model": "model", "system": "user"}


def _is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in _TRUTHY if value is not None else False


class GoogleProvider(Provider):
    """Serve Gemini models via Google's Generative Language API."""

    name = "google"

    def infer(self, resolved: ResolvedModel, payload: dict[str, Any]) -> dict[str, Any]:
        client = self._client(resolved)
        model, contents, config = self._build_request(resolved, payload)
        try:
            resp = client.models.generate_content(model=model, contents=contents, config=config)
        except Exception as exc:  # noqa: BLE001 - normalize any SDK/API error
            raise ProviderError(f"Google (Gemini) request failed: {exc}") from exc
        return self._map_response(resp, model)

    # --- seam (patched in tests) ---------------------------------------------
    def _client(self, resolved: ResolvedModel) -> Any:
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise ProviderError(
                "The 'google-genai' package is required for provider 'google'. "
                "Install it with:  pip install 'modelrack[google]'"
            ) from exc

        cfg = resolved.merged_config
        # --- Vertex AI mode (Google Cloud) --------------------------------------
        # Uses Application Default Credentials (ADC) — the service-account key is
        # referenced by GOOGLE_APPLICATION_CREDENTIALS (or workload identity), never
        # stored by modelrack. project/location are NOT secrets and may live in config
        # or the env (GOOGLE_CLOUD_PROJECT / GOOGLE_CLOUD_LOCATION). Enable via
        # `vertexai: true` in config or GOOGLE_GENAI_USE_VERTEXAI=true.
        use_vertex = _is_truthy(cfg.get("vertexai")) or _is_truthy(
            os.environ.get("GOOGLE_GENAI_USE_VERTEXAI")
        )
        if use_vertex:
            kwargs: dict[str, Any] = {"vertexai": True}
            project = cfg.get("project") or os.environ.get("GOOGLE_CLOUD_PROJECT")
            location = cfg.get("location") or os.environ.get("GOOGLE_CLOUD_LOCATION")
            if project:
                kwargs["project"] = project
            if location:
                kwargs["location"] = location
            return genai.Client(**kwargs)

        # --- API-key mode (AI Studio) ------------------------------------------
        key = self.resolve_api_key(resolved, _DEFAULT_KEY_ENV)
        # No key set → let the SDK resolve credentials from its own chain
        # (GEMINI_API_KEY / GOOGLE_API_KEY / Application Default Credentials).
        return genai.Client(api_key=key) if key else genai.Client()

    # --- pure helpers (unit-tested) ------------------------------------------
    @staticmethod
    def _build_request(
        resolved: ResolvedModel, payload: dict[str, Any]
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        messages = payload.get("messages")
        if not messages:
            raise ProviderError("payload.messages is required for this Google model")

        contents: list[dict[str, Any]] = []
        for msg in messages:
            role = _ROLE_MAP.get(msg.get("role", "user"), "user")
            content = msg.get("content")
            if isinstance(content, list):
                parts = content  # already provider-native parts (e.g. inline images)
            else:
                parts = [{"text": content if isinstance(content, str) else str(content)}]
            contents.append({"role": role, "parts": parts})

        config: dict[str, Any] = {"max_output_tokens": int(payload.get("max_tokens", 1024))}
        if payload.get("system"):
            config["system_instruction"] = payload["system"]
        # provider_params supplies native fields; normalized core fields win on overlap.
        extra = payload.get("provider_params")
        if isinstance(extra, dict):
            config = {**extra, **config}

        model = resolved.api_model_id or _DEFAULT_MODEL
        return model, contents, config

    @staticmethod
    def _map_response(resp: Any, model: str) -> dict[str, Any]:
        text = ""
        stop_reason = None
        candidates = getattr(resp, "candidates", None) or []
        if candidates:
            first = candidates[0]
            content = getattr(first, "content", None)
            for part in (getattr(content, "parts", None) or []):
                piece = getattr(part, "text", None)
                if piece:
                    text += piece
            finish = getattr(first, "finish_reason", None)
            stop_reason = getattr(finish, "name", finish)
        if not text:  # fall back to the SDK's convenience accessor
            text = getattr(resp, "text", "") or ""

        usage = getattr(resp, "usage_metadata", None)
        return {
            "text": text.strip() if isinstance(text, str) else text,
            "model": model,
            "stop_reason": str(stop_reason) if stop_reason is not None else None,
            "usage": {
                "input_tokens": getattr(usage, "prompt_token_count", None) if usage else None,
                "output_tokens": getattr(usage, "candidates_token_count", None) if usage else None,
            },
        }
