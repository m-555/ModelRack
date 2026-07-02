"""Validation of model configs, runtime params, weights and venvs."""

from __future__ import annotations

import re
from typing import Any

from modelrack.resolver import _check_param
from modelrack.schemas.model_types import is_known_type
from modelrack.schemas.resolved_model import ResolvedModel
from modelrack.schemas.validation import ValidationResult
from modelrack.utils.paths import venv_dir, venv_exists

REQUIRED_TOP_LEVEL = ("model_id", "type")
_VERSION_RE = re.compile(r"^\d+(\.\d+){0,2}$")


class ConfigValidator:
    """Static validators; each returns a :class:`ValidationResult`."""

    def validate_model_config(self, config: dict[str, Any]) -> ValidationResult:
        """Check structural correctness of a raw config dict."""
        result = ValidationResult()

        for field in REQUIRED_TOP_LEVEL:
            if field not in config:
                result.add_error(f"Missing required top-level field: '{field}'")

        model_type = config.get("type")
        if model_type is not None and not is_known_type(model_type):
            result.add_warning(
                f"Unknown model type '{model_type}' (no default template/requirements). "
                "Register it with modelrack.register_type() if this is intentional."
            )

        backend = config.get("backend", "local")

        # param_schema entries must at least declare type + label.
        for name, spec in (config.get("param_schema") or {}).items():
            if not isinstance(spec, dict):
                result.add_error(f"param_schema['{name}'] must be a mapping")
                continue
            if "type" not in spec:
                result.add_error(f"param_schema['{name}'] missing required key 'type'")
            if "label" not in spec:
                result.add_warning(f"param_schema['{name}'] missing 'label' (UI display name)")

        # Server section (local models need a valid port).
        if backend == "local":
            server = config.get("server") or {}
            port = server.get("port")
            if port is None:
                result.add_error("Local model missing 'server.port'")
            elif not isinstance(port, int) or not (1 <= port <= 65535):
                result.add_error(f"server.port must be an int in 1-65535, got {port!r}")

            env = config.get("environment") or {}
            pyver = env.get("python_version")
            if pyver is not None and not _VERSION_RE.match(str(pyver)):
                result.add_error(f"environment.python_version invalid: {pyver!r}")

        return result

    def validate_runtime_params(
        self, params: dict[str, Any], param_schema: dict[str, Any]
    ) -> ValidationResult:
        """Validate params against a schema, collecting *all* violations."""
        result = ValidationResult()
        for key, value in params.items():
            spec = param_schema.get(key)
            if spec is None:
                result.add_warning(f"Param '{key}' is not in param_schema (passed through)")
                continue
            for msg in _check_param(key, value, spec):
                result.add_error(msg)
        return result

    def validate_weight_paths(self, resolved: ResolvedModel) -> ValidationResult:
        """Check that the main weight exists; warn on missing optional weights."""
        result = ValidationResult()
        if resolved.is_api:
            return result

        weights = (resolved.merged_config.get("weights") or {}) if resolved.merged_config else {}
        model_dir = resolved.model_dir
        if model_dir is None:
            result.add_error("Resolved model has no model_dir")
            return result

        for name, rel in weights.items():
            if rel is None:
                continue
            path = model_dir / rel
            if path.exists():
                continue
            if name == "main":
                result.add_error(f"Main weight missing: {path}")
            else:
                result.add_warning(f"Optional weight '{name}' missing: {path}")
        return result

    def validate_venv(self, resolved: ResolvedModel) -> ValidationResult:
        """Check the model's isolated .venv exists and has a Python interpreter."""
        result = ValidationResult()
        if resolved.is_api:
            return result
        model_dir = resolved.model_dir
        if model_dir is None:
            result.add_error("Resolved model has no model_dir")
            return result

        venv_path = (resolved.environment or {}).get("venv_path", ".venv")
        if venv_exists(model_dir, venv_path):
            return result

        if (model_dir / "requirements.txt").exists():
            result.add_warning(
                f".venv not found at {venv_dir(model_dir, venv_path)} - "
                f"run 'modelrack setup {resolved.model_id}' first."
            )
        else:
            result.add_warning(
                f".venv not found and no requirements.txt present for {resolved.model_id}."
            )
        return result

    def validate_all(self, resolved: ResolvedModel) -> ValidationResult:
        """Run config + weights + venv validators and merge the results."""
        result = ValidationResult()
        if resolved.merged_config:
            result.merge(self.validate_model_config(resolved.merged_config))
        result.merge(self.validate_weight_paths(resolved))
        result.merge(self.validate_venv(resolved))
        return result
