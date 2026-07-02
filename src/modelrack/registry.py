"""CRUD over ``registry.yaml`` — the master index of all models.

The registry file is auto-maintained by the hub and never edited by hand. Writes are
always atomic (write to ``registry.yaml.tmp`` then ``os.replace``) so a crash can
never corrupt the index.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from modelrack.exceptions import ModelAlreadyExistsError, ModelNotFoundError

REGISTRY_FILENAME = "registry.yaml"
REGISTRY_VERSION = "1.0"


class ModelRegistry:
    """Read/write access to ``<models_dir>/registry.yaml``."""

    def __init__(self, models_dir: Path) -> None:
        self.models_dir = Path(models_dir)
        self.registry_path = self.models_dir / REGISTRY_FILENAME

    # --- Loading / saving -----------------------------------------------------
    def _load(self) -> dict[str, Any]:
        if not self.registry_path.exists():
            return {"version": REGISTRY_VERSION, "models": {}}
        data = yaml.safe_load(self.registry_path.read_text(encoding="utf-8")) or {}
        data.setdefault("version", REGISTRY_VERSION)
        data.setdefault("models", {})
        if data["models"] is None:
            data["models"] = {}
        return data

    def _write(self, data: dict[str, Any] | None = None) -> None:
        """Atomically persist the registry (tmp file + ``os.replace``)."""
        if data is None:
            data = self._load()
        self.models_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.registry_path.with_suffix(self.registry_path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True)
        os.replace(tmp, self.registry_path)

    # --- Queries --------------------------------------------------------------
    def list_models(
        self,
        type: str | None = None,
        backend: str | None = None,
        tags: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return registry entries matching all given filters (AND logic).

        Never raises; returns ``[]`` when nothing matches. Each entry includes its
        ``model_id`` key for convenience.
        """
        models = self._load()["models"]
        out: list[dict[str, Any]] = []
        want_tags = set(tags) if tags else None
        for model_id, entry in models.items():
            if type is not None and entry.get("type") != type:
                continue
            if backend is not None and entry.get("backend") != backend:
                continue
            if want_tags is not None and not want_tags.issubset(set(entry.get("tags") or [])):
                continue
            out.append({**entry, "model_id": model_id})
        return out

    def get_model_entry(self, model_id: str) -> dict[str, Any]:
        """Return a single registry entry (with ``model_id``), or raise."""
        models = self._load()["models"]
        if model_id not in models:
            raise ModelNotFoundError(
                f"Model '{model_id}' is not registered. Run 'modelrack scan' or "
                f"'modelrack add {model_id} ...' first."
            )
        return {**models[model_id], "model_id": model_id}

    def exists(self, model_id: str) -> bool:
        return model_id in self._load()["models"]

    # --- Mutations ------------------------------------------------------------
    def add_model(
        self,
        model_id: str,
        type: str,
        backend: str,
        config_path: str | None = None,
        tags: list[str] | None = None,
        **extra: Any,
    ) -> None:
        """Register a new model. Raises if the id already exists."""
        data = self._load()
        if model_id in data["models"]:
            raise ModelAlreadyExistsError(f"Model '{model_id}' is already registered.")
        entry: dict[str, Any] = {
            "type": type,
            "backend": backend,
            "tags": tags or [],
            "added_at": datetime.now().isoformat(timespec="seconds"),
        }
        if config_path is not None:
            entry["config_path"] = config_path
        entry.update(extra)
        data["models"][model_id] = entry
        self._write(data)

    def remove_model(self, model_id: str) -> None:
        """Remove from the registry only. Never deletes files on disk."""
        data = self._load()
        if model_id not in data["models"]:
            raise ModelNotFoundError(f"Model '{model_id}' is not registered.")
        del data["models"][model_id]
        self._write(data)

    def update_model(self, model_id: str, **kwargs: Any) -> None:
        """Update registry-level metadata (not the model's config.yaml)."""
        data = self._load()
        if model_id not in data["models"]:
            raise ModelNotFoundError(f"Model '{model_id}' is not registered.")
        data["models"][model_id].update(kwargs)
        self._write(data)

    # --- Discovery ------------------------------------------------------------
    def scan_and_sync(self) -> dict[str, list[str]]:
        """Scan MODELS_DIR for model folders and register any that are new.

        A "model folder" is any immediate subdirectory containing a ``config.yaml``.
        Returns a report: ``{"added", "already_registered", "missing_config"}``.
        """
        data = self._load()
        registered = data["models"]
        report: dict[str, list[str]] = {
            "added": [],
            "already_registered": [],
            "missing_config": [],
        }
        if not self.models_dir.exists():
            return report

        for child in sorted(self.models_dir.iterdir()):
            if not child.is_dir() or child.name.startswith("."):
                continue
            config_file = child / "config.yaml"
            model_id = child.name
            if not config_file.exists():
                report["missing_config"].append(model_id)
                continue
            if model_id in registered:
                report["already_registered"].append(model_id)
                continue

            cfg = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
            registered[model_id] = {
                "type": cfg.get("type", "unknown"),
                "backend": cfg.get("backend", "local"),
                "config_path": f"{model_id}/config.yaml",
                "tags": cfg.get("tags") or (cfg.get("meta") or {}).get("tags") or [],
                "added_at": datetime.now().isoformat(timespec="seconds"),
            }
            report["added"].append(model_id)

        if report["added"]:
            self._write(data)
        return report
