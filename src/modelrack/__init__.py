"""modelrack — centralized model registry, config pipeline & process manager.

Typical use::

    from modelrack import ModelRack

    hub = ModelRack()                       # reads MODELS_DIR from env
    hub = ModelRack(models_dir="/models")   # or pass explicitly

    model = hub.resolve("qwen-image", runtime_params={"num_inference_steps": 30})
    hub.setup("qwen-image")
    result = hub.infer("qwen-image", {"prompt": "a red fox in snow"})
"""

from __future__ import annotations

import builtins
import logging
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from modelrack import exceptions
from modelrack.config import configure_logging, load_settings
from modelrack.exceptions import ModelRackError
from modelrack.inference_client import InferenceClient
from modelrack.process_manager import ProcessManager
from modelrack.registry import ModelRegistry
from modelrack.resolver import ModelResolver
from modelrack.schemas import (
    Backend,
    ResolvedModel,
    ServerProcess,
    ValidationResult,
    register_type,
)
from modelrack.validator import ConfigValidator
from modelrack.watcher import ConfigWatcher

__version__ = "0.1.0"

__all__ = [
    "ModelRack",
    "ModelRegistry",
    "ModelResolver",
    "ConfigValidator",
    "ProcessManager",
    "InferenceClient",
    "ConfigWatcher",
    "ResolvedModel",
    "ServerProcess",
    "ValidationResult",
    "Backend",
    "register_type",
    "exceptions",
    "ModelRackError",
    "__version__",
]


class ModelRack:
    """The single public entry point wiring registry, resolver, validator,
    process manager and inference client together."""

    def __init__(self, models_dir: str | Path | None = None) -> None:
        self.settings = load_settings(models_dir)
        configure_logging(self.settings.log_level)
        self.models_dir = self.settings.require_models_dir()

        self.registry = ModelRegistry(self.models_dir)
        self.resolver = ModelResolver(self.models_dir, self.registry)
        self.validator = ConfigValidator()
        self.processes = ProcessManager(
            self.models_dir,
            state_file=self.settings.state_file,
            resolver=self.resolver,
        )
        self.client = InferenceClient(self.processes)
        self._watcher: ConfigWatcher | None = None

    # --- Config resolution ----------------------------------------------------
    def resolve(
        self,
        model_id: str,
        app_overrides: dict[str, Any] | None = None,
        runtime_params: dict[str, Any] | None = None,
    ) -> ResolvedModel:
        return self.resolver.resolve(model_id, app_overrides, runtime_params)

    def resolve_from_app(
        self,
        model_id: str,
        app_overrides_path: str | Path,
        runtime_params: dict[str, Any] | None = None,
    ) -> ResolvedModel:
        return self.resolver.resolve_from_app(model_id, app_overrides_path, runtime_params)

    # --- Listing / schema -----------------------------------------------------
    # NB: this class defines a method named ``list``, which shadows the builtin
    # inside the class body — so annotations use ``builtins.list`` explicitly.
    def list(
        self,
        type: str | None = None,
        backend: str | None = None,
        tags: builtins.list[str] | None = None,
    ) -> builtins.list[dict[str, Any]]:
        return self.registry.list_models(type=type, backend=backend, tags=tags)

    def schema(self, model_id: str) -> dict[str, Any]:
        return self.resolver.get_param_schema(model_id)

    def scan(self) -> dict[str, builtins.list[str]]:
        return self.registry.scan_and_sync()

    # --- Validation -----------------------------------------------------------
    def validate(self, model_id: str, check_venv: bool = True) -> ValidationResult:
        resolved = self.resolver.resolve(model_id)
        if check_venv:
            return self.validator.validate_all(resolved)
        result = self.validator.validate_model_config(resolved.merged_config)
        return result.merge(self.validator.validate_weight_paths(resolved))

    # --- Process management ---------------------------------------------------
    def setup(self, model_id: str, force: bool = False) -> None:
        self.processes.setup(model_id, force=force)

    def start(self, model_id: str) -> ServerProcess:
        return self.processes.start(model_id)

    def stop(self, model_id: str, graceful: bool = True) -> None:
        self.processes.stop(model_id, graceful=graceful)

    def restart(self, model_id: str) -> ServerProcess:
        return self.processes.restart(model_id)

    def status(self, model_id: str | None = None) -> builtins.list[ServerProcess]:
        return self.processes.status(model_id)

    def is_running(self, model_id: str) -> bool:
        return self.processes.is_running(model_id)

    # --- Inference ------------------------------------------------------------
    def infer(
        self,
        model_id: str,
        payload: dict[str, Any],
        auto_start: bool = True,
        timeout: int = 300,
    ) -> dict[str, Any]:
        # API models run in-process via a provider adapter; local models route over
        # HTTP to their subprocess server. Both return the {success,data,error} envelope.
        if self._is_api_model(model_id):
            return self._api_infer(model_id, payload)
        return self.client.infer(model_id, payload, auto_start=auto_start, timeout=timeout)

    def stream_infer(
        self,
        model_id: str,
        payload: dict[str, Any],
        auto_start: bool = True,
        timeout: int = 300,
    ) -> Iterator[str]:
        """Yield generated text chunks. Local models stream token-by-token from their
        ``/infer_stream`` endpoint; API models yield the full result once (provider
        streaming is a future addition)."""
        if self._is_api_model(model_id):
            data = self._api_infer(model_id, payload).get("data") or {}
            yield (data.get("text") or "") if isinstance(data, dict) else str(data)
            return
        yield from self.client.stream_infer(
            model_id, payload, auto_start=auto_start, timeout=timeout
        )

    def _is_api_model(self, model_id: str) -> bool:
        try:
            return self.registry.get_model_entry(model_id).get("backend") == "api"
        except Exception:  # noqa: BLE001 - unknown/unregistered -> treat as local
            return False

    def _api_infer(self, model_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        from modelrack.providers import get_provider

        resolved = self.resolver.resolve(model_id)
        provider = get_provider(resolved.provider)
        merged = {**resolved.merged_config.get("defaults", {}), **payload}
        data = provider.infer(resolved, merged)
        return {"success": True, "data": data, "error": None}

    def unload(self, model_id: str) -> dict[str, Any]:
        return self.client.unload(model_id)

    def info(self, model_id: str) -> dict[str, Any]:
        return self.client.info(model_id)

    # --- Hot reload -----------------------------------------------------------
    def watch(self, on_change: Callable[[str, dict[str, Any]], None]) -> ConfigWatcher:
        watcher = ConfigWatcher(self.models_dir, on_change)
        watcher.start()
        self._watcher = watcher
        return watcher

    def stop_watching(self) -> None:
        if self._watcher is not None:
            self._watcher.stop()
            self._watcher = None


logging.getLogger("modelrack").addHandler(logging.NullHandler())
