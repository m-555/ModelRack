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
