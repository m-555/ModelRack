"""Three-layer config resolution: base config -> app overrides -> runtime params.

The resolver produces a :class:`ResolvedModel` — the single object every consumer
(CLI, API, Python callers) works with. Merge order is *later wins*, with deep merge
on nested dicts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from modelrack.exceptions import (
    AppOverridesNotFoundError,
    ConfigNotFoundError,
    ParamValidationError,
    WeightNotFoundError,
)
from modelrack.registry import ModelRegistry
from modelrack.schemas.model_types import Backend
from modelrack.schemas.resolved_model import ResolvedModel
from modelrack.utils.merge import deep_merge

logger = logging.getLogger("modelrack")


class ModelResolver:
    """Resolve a registered model id into a fully merged :class:`ResolvedModel`."""

    def __init__(self, models_dir: Path, registry: ModelRegistry | None = None) -> None:
        self.models_dir = Path(models_dir)
        self.registry = registry or ModelRegistry(self.models_dir)

    # --- Public API -----------------------------------------------------------
    def resolve(
        self,
        model_id: str,
        app_overrides: dict[str, Any] | None = None,
        runtime_params: dict[str, Any] | None = None,
    ) -> ResolvedModel:
        """Resolve a model, merging base config, app overrides and runtime params."""
        entry = self.registry.get_model_entry(model_id)
        backend = entry.get("backend", "local")

        if backend == Backend.API.value:
            return self._resolve_api(model_id, entry, app_overrides, runtime_params)
        return self._resolve_local(model_id, entry, app_overrides, runtime_params)

    def resolve_from_app(
        self,
        model_id: str,
        app_overrides_path: str | Path,
        runtime_params: dict[str, Any] | None = None,
    ) -> ResolvedModel:
        """Load app overrides from a YAML file, then resolve.

        The overrides file may be keyed as ``models: {<model_id>: {...}}`` (the app
        convention) or be the override dict for this model directly.
        """
        path = Path(app_overrides_path)
        if not path.exists():
            raise AppOverridesNotFoundError(f"App overrides file not found: {path}")
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        overrides = data.get("models", {}).get(model_id) if "models" in data else data
        return self.resolve(model_id, app_overrides=overrides or {}, runtime_params=runtime_params)

    def get_param_schema(self, model_id: str) -> dict[str, Any]:
        """Return the model's ``param_schema`` from base config (never overridden)."""
        config = self._load_config(model_id)
        return config.get("param_schema", {})

    def list_available(
        self, backend: str | None = None, type: str | None = None
    ) -> list[ResolvedModel]:
        """Resolve every matching model with defaults only.

        Models whose config.yaml is missing or malformed are skipped with a warning,
        never raised.
        """
        out: list[ResolvedModel] = []
        for entry in self.registry.list_models(type=type, backend=backend):
            try:
                out.append(self.resolve(entry["model_id"]))
            except Exception as exc:  # noqa: BLE001 - resilience is the point here
                logger.warning("Skipping %s: %s", entry["model_id"], exc)
        return out

    # --- Local resolution -----------------------------------------------------
    def _resolve_local(
        self,
        model_id: str,
        entry: dict[str, Any],
        app_overrides: dict[str, Any] | None,
        runtime_params: dict[str, Any] | None,
    ) -> ResolvedModel:
        model_dir = self.models_dir / model_id
        config = self._load_config(model_id)

        # Layer 1 -> 2: base config <- app overrides (deep merge).
        merged = deep_merge(config, app_overrides or {})

        param_schema = config.get("param_schema", {})

        # Layer 3: runtime params (validated) win over defaults.
        defaults = dict(merged.get("defaults", {}))
        if runtime_params:
            self._validate_runtime_params(model_id, runtime_params, param_schema)
            defaults = deep_merge(defaults, runtime_params)
        merged["defaults"] = defaults

        weight_paths = self._resolve_weights(model_id, model_dir, config.get("weights", {}))

        server = merged.get("server", {})
        port = server.get("port")
        host = server.get("host", "127.0.0.1")
        server_url = f"http://{host}:{port}" if port else None

        return ResolvedModel(
            model_id=model_id,
            display_name=config.get("display_name", model_id),
            type=config.get("type", entry.get("type", "unknown")),
            backend="local",
            model_dir=model_dir,
            weight_paths=weight_paths,
            server_url=server_url,
            server_port=port,
            merged_config=merged,
            param_schema=param_schema,
            load_hints=merged.get("load_hints", {}),
            hardware=merged.get("hardware", {}),
            environment=merged.get("environment", {}),
            serving=merged.get("serving", {}),
            tags=entry.get("tags", []) or config.get("tags", []),
        )

    def _resolve_api(
        self,
        model_id: str,
        entry: dict[str, Any],
        app_overrides: dict[str, Any] | None,
        runtime_params: dict[str, Any] | None,
    ) -> ResolvedModel:
        # API models may have an optional config.yaml for defaults/param_schema.
        config = self._load_config(model_id, required=False)
        merged = deep_merge(config, app_overrides or {})
        param_schema = config.get("param_schema", {})

        defaults = dict(merged.get("defaults", {}))
        if runtime_params:
            self._validate_runtime_params(model_id, runtime_params, param_schema)
            defaults = deep_merge(defaults, runtime_params)
        merged["defaults"] = defaults

        return ResolvedModel(
            model_id=model_id,
            display_name=config.get("display_name", model_id),
            type=config.get("type", entry.get("type", "language")),
            backend="api",
            provider=entry.get("provider"),
            api_model_id=entry.get("api_model_id") or entry.get("model_id"),
            merged_config=merged,
            param_schema=param_schema,
            load_hints=merged.get("load_hints", {}),
            hardware=merged.get("hardware", {}),
            environment=merged.get("environment", {}),
            serving=merged.get("serving", {}),
            tags=entry.get("tags", []) or config.get("tags", []),
        )

    # --- Helpers --------------------------------------------------------------
    def _load_config(self, model_id: str, required: bool = True) -> dict[str, Any]:
        config_path = self.models_dir / model_id / "config.yaml"
        if not config_path.exists():
            if required:
                raise ConfigNotFoundError(f"config.yaml not found for '{model_id}': {config_path}")
            return {}
        return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    def _resolve_weights(
        self, model_id: str, model_dir: Path, weights: dict[str, Any]
    ) -> dict[str, Path]:
        """Resolve weight paths to absolute. Missing optional weights are skipped."""
        resolved: dict[str, Path] = {}
        for name, rel in weights.items():
            if rel is None:
                continue
            abs_path = (model_dir / rel).resolve()
            if name == "main" and not abs_path.exists():
                raise WeightNotFoundError(f"Main weight for '{model_id}' not found: {abs_path}")
            if not abs_path.exists():
                logger.warning("Optional weight '%s' for %s missing: %s", name, model_id, abs_path)
                continue
            resolved[name] = abs_path
        return resolved

    def _validate_runtime_params(
        self, model_id: str, params: dict[str, Any], schema: dict[str, Any]
    ) -> None:
        """Validate runtime params against param_schema. Collect all violations."""
        errors: list[str] = []
        for key, value in params.items():
            spec = schema.get(key)
            if spec is None:
                continue  # unknown params pass through (forward-compat for app UIs)
            errors.extend(_check_param(key, value, spec))
        if errors:
            raise ParamValidationError(
                f"Invalid runtime params for '{model_id}':\n  - " + "\n  - ".join(errors)
            )


def _check_param(key: str, value: Any, spec: dict[str, Any]) -> list[str]:
    """Return a list of violation messages for one param against its spec."""
    errors: list[str] = []
    ptype = spec.get("type")
    is_bool = isinstance(value, bool)
    is_number = isinstance(value, (int, float)) and not is_bool

    # Type checks (bool is intentionally NOT accepted as int/float).
    if ptype == "int" and (is_bool or not isinstance(value, int)):
        errors.append(f"{key}: expected int, got {type(value).__name__}")
    elif ptype == "float" and not is_number:
        errors.append(f"{key}: expected float, got {type(value).__name__}")
    elif ptype == "bool" and not is_bool:
        errors.append(f"{key}: expected bool, got {type(value).__name__}")
    elif ptype == "str" and not isinstance(value, str):
        errors.append(f"{key}: expected str, got {type(value).__name__}")

    # Range checks (only meaningful for numbers).
    if is_number:
        if "min" in spec and value < spec["min"]:
            errors.append(f"{key}: {value} below min {spec['min']}")
        if "max" in spec and value > spec["max"]:
            errors.append(f"{key}: {value} above max {spec['max']}")

    # Enumerated options.
    if "options" in spec and value not in spec["options"]:
        errors.append(f"{key}: {value!r} not in allowed options {spec['options']}")

    return errors
