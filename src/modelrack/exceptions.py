"""Custom exception hierarchy for modelrack.

Every error raised by the library derives from :class:`ModelRackError`, so callers
can catch the whole family with a single ``except ModelRackError``.
"""

from __future__ import annotations


class ModelRackError(Exception):
    """Base class for all modelrack errors."""


class ModelsDirNotConfiguredError(ModelRackError):
    """Raised when MODELS_DIR is not set or does not exist."""


class ModelNotFoundError(ModelRackError):
    """Raised when a model id is not present in the registry."""


class ModelAlreadyExistsError(ModelRackError):
    """Raised when adding a model id that is already registered."""


class ConfigNotFoundError(ModelRackError):
    """Raised when a model's config.yaml cannot be located."""


class ConfigValidationError(ModelRackError):
    """Raised when a config or set of runtime params fails validation."""


class WeightNotFoundError(ModelRackError):
    """Raised when a model's required (main) weight file is missing on disk."""


class AppOverridesNotFoundError(ModelRackError):
    """Raised when an app overrides YAML file cannot be found."""


class ParamValidationError(ModelRackError):
    """Raised when runtime params violate a model's param_schema."""


class PortConflictError(ModelRackError):
    """Raised when a model's configured port is already in use."""


class ServerStartupError(ModelRackError):
    """Raised when an inference server fails to become healthy in time."""


class ServerNotRunningError(ModelRackError):
    """Raised when an operation requires a running server but none is running."""


class InferenceError(ModelRackError):
    """Raised when a model server returns a non-success inference response."""


class UvNotFoundError(ModelRackError):
    """Raised when the ``uv`` executable is not found on PATH."""


class SetupNotCompletedError(ModelRackError):
    """Raised when starting a model whose .venv has not been set up yet."""
