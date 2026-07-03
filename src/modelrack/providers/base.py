"""Base class for in-process API-model providers.

API models (`backend: api`) run **in the hub process**, not as a subprocess — a
provider translates modelrack's normalized payload into a cloud provider's native
SDK call and normalizes the response back. Provider SDKs are light, pure-Python, and
lazily imported, so the base install stays free of heavy dependencies.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

from modelrack.exceptions import ModelRackError
from modelrack.schemas.resolved_model import ResolvedModel


class ProviderError(ModelRackError):
    """An API-provider call failed (missing SDK, missing credential, or an API error)."""


class Provider(ABC):
    """A cloud API provider (Anthropic, OpenAI, Google, …)."""

    #: registry key, e.g. "anthropic"
    name: str = ""

    @abstractmethod
    def infer(self, resolved: ResolvedModel, payload: dict[str, Any]) -> dict[str, Any]:
        """Run one inference and return normalized ``data`` (e.g. ``{"text", "usage", ...}``).

        ``payload`` is the model's ``defaults`` merged with the request params. Raise
        :class:`ProviderError` on failure.
        """

    @staticmethod
    def resolve_api_key(resolved: ResolvedModel, default_env: str) -> str | None:
        """Return the API key from the env var named by the model's ``api_key_env``
        (default ``default_env``), or ``None`` if unset.

        modelrack stores a *reference* (the env-var name) in config, never the secret.
        ``None`` is a valid result — a provider may still let its SDK resolve credentials
        from its own chain (env vars, CLI login profiles, workload identity, …).
        """
        env_name = resolved.merged_config.get("api_key_env", default_env)
        return os.environ.get(env_name)
