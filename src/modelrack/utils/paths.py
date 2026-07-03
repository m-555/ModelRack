"""Filesystem path helpers (cross-platform venv resolution, etc.)."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def expand(path: str | os.PathLike[str]) -> Path:
    """Expand ``~`` and environment variables, returning an absolute Path."""
    return Path(os.path.expandvars(os.path.expanduser(str(path)))).resolve()


def venv_dir(model_dir: Path, venv_path: str = ".venv") -> Path:
    """Absolute path to a model's virtual environment directory."""
    p = Path(venv_path)
    return p if p.is_absolute() else (model_dir / p)


def venv_python(model_dir: Path, venv_path: str = ".venv") -> Path:
    """Path to the venv's Python interpreter, correct for the current OS.

    Windows venvs use ``Scripts/python.exe``; POSIX venvs use ``bin/python``.
    """
    vdir = venv_dir(model_dir, venv_path)
    if sys.platform == "win32":
        return vdir / "Scripts" / "python.exe"
    return vdir / "bin" / "python"


def venv_exists(model_dir: Path, venv_path: str = ".venv") -> bool:
    """True if the venv directory and its interpreter both exist."""
    return venv_dir(model_dir, venv_path).is_dir() and venv_python(model_dir, venv_path).exists()


def shared_venvs_root(models_dir: Path) -> Path:
    """Directory that holds venvs shared across models (``environment.shared_venv``)."""
    return Path(models_dir) / "_shared_venvs"


def resolve_venv_dir(models_dir: Path, model_dir: Path, environment: dict) -> Path:
    """Absolute venv directory for a model, honoring a shared venv.

    ``environment.shared_venv: <name>`` resolves to
    ``<models_dir>/_shared_venvs/<name>`` — one venv reused by every model that names it
    (build heavy deps like torch once, share across compatible models). Otherwise falls
    back to ``environment.venv_path`` (default ``.venv``), resolved relative to the model
    directory (or an absolute path).
    """
    shared = environment.get("shared_venv")
    if shared:
        return shared_venvs_root(models_dir) / str(shared)
    return venv_dir(model_dir, environment.get("venv_path", ".venv"))


def resolve_venv_python(models_dir: Path, model_dir: Path, environment: dict) -> Path:
    """Python interpreter for a model's (possibly shared) venv."""
    vdir = resolve_venv_dir(models_dir, model_dir, environment)
    if sys.platform == "win32":
        return vdir / "Scripts" / "python.exe"
    return vdir / "bin" / "python"


def resolve_venv_exists(models_dir: Path, model_dir: Path, environment: dict) -> bool:
    """True if the model's (possibly shared) venv and its interpreter both exist."""
    vdir = resolve_venv_dir(models_dir, model_dir, environment)
    return vdir.is_dir() and resolve_venv_python(models_dir, model_dir, environment).exists()


def venv_python_version(venv_directory: Path) -> str | None:
    """Return the ``X.Y.Z`` recorded in the venv's ``pyvenv.cfg`` (None if unknown)."""
    cfg = Path(venv_directory) / "pyvenv.cfg"
    if not cfg.exists():
        return None
    for line in cfg.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.strip().lower().startswith("version"):
            return line.split("=", 1)[1].strip()
    return None
