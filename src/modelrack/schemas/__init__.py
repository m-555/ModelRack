"""Data schemas used across modelrack."""

from __future__ import annotations

from modelrack.schemas.model_types import (
    Backend,
    TypeSpec,
    get_type_spec,
    is_known_type,
    known_types,
    register_type,
    requirements_for,
    template_for,
)
from modelrack.schemas.resolved_model import ResolvedModel
from modelrack.schemas.server_process import ServerProcess
from modelrack.schemas.validation import ValidationResult

__all__ = [
    "Backend",
    "TypeSpec",
    "ResolvedModel",
    "ServerProcess",
    "ValidationResult",
    "get_type_spec",
    "is_known_type",
    "known_types",
    "register_type",
    "requirements_for",
    "template_for",
]
