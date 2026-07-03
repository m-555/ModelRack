"""Unit tests for shared-venv resolution (environment.shared_venv)."""

from __future__ import annotations

import sys
from pathlib import Path

from modelrack.utils.paths import (
    resolve_venv_dir,
    resolve_venv_exists,
    resolve_venv_python,
    shared_venvs_root,
    venv_python_version,
)


def _expected_python(vdir: Path) -> Path:
    return vdir / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def test_default_per_model_venv():
    models = Path("/models")
    model_dir = models / "my-model"
    env: dict = {}
    assert resolve_venv_dir(models, model_dir, env) == model_dir / ".venv"


def test_custom_venv_path():
    models = Path("/models")
    model_dir = models / "my-model"
    env = {"venv_path": "env311"}
    assert resolve_venv_dir(models, model_dir, env) == model_dir / "env311"


def test_shared_venv_resolves_under_models_dir():
    models = Path("/models")
    model_dir = models / "my-model"
    env = {"shared_venv": "torch-cuda"}
    assert resolve_venv_dir(models, model_dir, env) == shared_venvs_root(models) / "torch-cuda"


def test_shared_venv_is_shared_across_models():
    models = Path("/models")
    env = {"shared_venv": "torch-cuda"}
    a = resolve_venv_dir(models, models / "model-a", env)
    b = resolve_venv_dir(models, models / "model-b", env)
    assert a == b  # both point at the one shared venv


def test_shared_venv_takes_precedence_over_venv_path():
    models = Path("/models")
    model_dir = models / "m"
    env = {"venv_path": ".venv", "shared_venv": "shared"}
    assert resolve_venv_dir(models, model_dir, env) == shared_venvs_root(models) / "shared"


def test_resolve_venv_python_location():
    models = Path("/models")
    model_dir = models / "m"
    env = {"shared_venv": "s"}
    vdir = resolve_venv_dir(models, model_dir, env)
    assert resolve_venv_python(models, model_dir, env) == _expected_python(vdir)


def test_resolve_venv_exists(tmp_path):
    models = tmp_path
    model_dir = models / "m"
    env = {"shared_venv": "s"}
    assert resolve_venv_exists(models, model_dir, env) is False

    vdir = resolve_venv_dir(models, model_dir, env)
    py = _expected_python(vdir)
    py.parent.mkdir(parents=True)
    py.write_text("")
    assert resolve_venv_exists(models, model_dir, env) is True


def test_venv_python_version_reads_pyvenv_cfg(tmp_path):
    (tmp_path / "pyvenv.cfg").write_text("home = /usr\nversion = 3.11.9\n", encoding="utf-8")
    assert venv_python_version(tmp_path) == "3.11.9"


def test_venv_python_version_missing(tmp_path):
    assert venv_python_version(tmp_path) is None
