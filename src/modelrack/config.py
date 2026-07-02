"""Environment-driven settings for modelrack.

All configuration is resolved from environment variables (optionally loaded from a
``.env`` file via python-dotenv). ``MODELS_DIR`` is the only required setting.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from modelrack.exceptions import ModelsDirNotConfiguredError
from modelrack.utils.paths import expand

DEFAULT_HUB_PORT = 7777
DEFAULT_STATE_FILE = "~/.modelrack/processes.json"

# Reserved port range for per-model inference servers (see config.yaml `server.port`).
MODEL_PORT_RANGE = range(7800, 7900)

_TRUE = {"1", "true", "yes", "on"}


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in _TRUE


@dataclass
class Settings:
    """Resolved runtime settings."""

    models_dir: Path | None
    hub_port: int
    log_level: str
    auto_start: bool
    state_file: str

    def require_models_dir(self) -> Path:
        """Return the models directory, raising a helpful error if unusable."""
        if self.models_dir is None:
            raise ModelsDirNotConfiguredError(
                "MODELS_DIR is not set. Point it at your models directory, e.g.\n"
                "    export MODELS_DIR=/path/to/models    (Linux/macOS)\n"
                '    $env:MODELS_DIR = "C:\\path\\to\\models"   (Windows PowerShell)'
            )
        if not self.models_dir.exists():
            raise ModelsDirNotConfiguredError(
                f"MODELS_DIR does not exist: {self.models_dir}\n"
                "Create it or point MODELS_DIR at an existing directory."
            )
        return self.models_dir


def load_settings(models_dir: str | os.PathLike[str] | None = None) -> Settings:
    """Load settings from the environment.

    ``models_dir`` (if given) overrides the ``MODELS_DIR`` env var.
    """
    load_dotenv(override=False)

    raw_dir = models_dir if models_dir is not None else os.environ.get("MODELS_DIR")
    resolved_dir = expand(raw_dir) if raw_dir else None

    port_raw = os.environ.get("MODELRACK_PORT")
    try:
        hub_port = int(port_raw) if port_raw else DEFAULT_HUB_PORT
    except ValueError:
        hub_port = DEFAULT_HUB_PORT

    return Settings(
        models_dir=resolved_dir,
        hub_port=hub_port,
        log_level=os.environ.get("MODELRACK_LOG_LEVEL", "INFO").upper(),
        auto_start=_env_bool("MODELRACK_AUTO_START", True),
        state_file=os.environ.get("MODELRACK_STATE_FILE", DEFAULT_STATE_FILE),
    )


def configure_logging(level: str) -> None:
    """Configure the ``modelrack`` logger once, idempotently."""
    logger = logging.getLogger("modelrack")
    if logger.handlers:
        logger.setLevel(level)
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s modelrack: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False


logger = logging.getLogger("modelrack")
