"""The fully-resolved view of a model, returned by the resolver."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class ResolvedModel:
    """A model with all three config layers merged and paths resolved.

    Local-only fields are ``None`` for API models and vice-versa.
    """

    # --- Identity -------------------------------------------------------------
    model_id: str
    display_name: str
    type: str
    backend: str  # "local" | "api"

    # --- Always present -------------------------------------------------------
    merged_config: dict[str, Any]
    param_schema: dict[str, Any]
    load_hints: dict[str, Any]
    hardware: dict[str, Any]
    environment: dict[str, Any]
    serving: dict[str, Any]
    tags: list[str] = field(default_factory=list)
    resolved_at: datetime = field(default_factory=datetime.now)

    # --- Local-only (None for API models) ------------------------------------
    model_dir: Path | None = None
    weight_paths: dict[str, Path] | None = None
    server_url: str | None = None
    server_port: int | None = None

    # --- API-only (None for local models) ------------------------------------
    provider: str | None = None
    api_model_id: str | None = None

    @property
    def is_local(self) -> bool:
        return self.backend == "local"

    @property
    def is_api(self) -> bool:
        return self.backend == "api"

    def to_dict(self) -> dict[str, Any]:
        """JSON/YAML-friendly representation (Paths -> str, datetime -> isoformat)."""

        def _p(v: Path | None) -> str | None:
            return str(v) if v is not None else None

        return {
            "model_id": self.model_id,
            "display_name": self.display_name,
            "type": self.type,
            "backend": self.backend,
            "model_dir": _p(self.model_dir),
            "weight_paths": (
                {k: str(v) for k, v in self.weight_paths.items()}
                if self.weight_paths is not None
                else None
            ),
            "provider": self.provider,
            "api_model_id": self.api_model_id,
            "server_url": self.server_url,
            "server_port": self.server_port,
            "merged_config": self.merged_config,
            "param_schema": self.param_schema,
            "load_hints": self.load_hints,
            "hardware": self.hardware,
            "environment": self.environment,
            "serving": self.serving,
            "tags": self.tags,
            "resolved_at": self.resolved_at.isoformat(),
        }
