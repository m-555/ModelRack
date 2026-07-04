"""Registry of API providers, keyed by the config's ``provider`` name."""

from __future__ import annotations

from modelrack.providers.anthropic_provider import AnthropicProvider
from modelrack.providers.base import Provider, ProviderError
from modelrack.providers.google_provider import GoogleProvider

_PROVIDERS: dict[str, type[Provider]] = {
    "anthropic": AnthropicProvider,
    "google": GoogleProvider,
}
_INSTANCES: dict[str, Provider] = {}


def register_provider(name: str, cls: type[Provider]) -> None:
    """Register (or replace) an API provider under ``name`` (e.g. "openai")."""
    key = name.lower()
    _PROVIDERS[key] = cls
    _INSTANCES.pop(key, None)


def get_provider(name: str | None) -> Provider:
    """Return a cached provider instance for ``name`` (from a model's registry entry)."""
    if not name:
        raise ProviderError("API model has no 'provider' set in its registry entry.")
    key = name.lower()
    cls = _PROVIDERS.get(key)
    if cls is None:
        raise ProviderError(f"Unknown API provider {name!r}. Known: {sorted(_PROVIDERS)}")
    if key not in _INSTANCES:
        _INSTANCES[key] = cls()
    return _INSTANCES[key]
