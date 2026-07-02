"""Tests for ModelRegistry."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from modelrack.exceptions import ModelAlreadyExistsError, ModelNotFoundError
from modelrack.registry import ModelRegistry


def test_list_models_no_filter(registry: ModelRegistry):
    ids = {m["model_id"] for m in registry.list_models()}
    assert ids == {"demo-image", "demo-llm"}


def test_list_models_by_type(registry: ModelRegistry):
    rows = registry.list_models(type="language")
    assert [m["model_id"] for m in rows] == ["demo-llm"]


def test_list_models_by_backend(registry: ModelRegistry):
    assert len(registry.list_models(backend="local")) == 2
    assert registry.list_models(backend="api") == []


def test_list_models_by_tags_and_logic(registry: ModelRegistry):
    assert [m["model_id"] for m in registry.list_models(tags=["image", "diffusion"])] == [
        "demo-image"
    ]
    assert registry.list_models(tags=["image", "nonexistent"]) == []


def test_get_add_remove_round_trip(registry: ModelRegistry):
    registry.add_model("new-model", type="tts", backend="local", tags=["audio"])
    entry = registry.get_model_entry("new-model")
    assert entry["type"] == "tts"
    assert entry["backend"] == "local"
    registry.remove_model("new-model")
    assert not registry.exists("new-model")


def test_add_duplicate_raises(registry: ModelRegistry):
    with pytest.raises(ModelAlreadyExistsError):
        registry.add_model("demo-image", type="image_generation", backend="local")


def test_get_missing_raises(registry: ModelRegistry):
    with pytest.raises(ModelNotFoundError):
        registry.get_model_entry("does-not-exist")


def test_remove_missing_raises(registry: ModelRegistry):
    with pytest.raises(ModelNotFoundError):
        registry.remove_model("does-not-exist")


def test_update_model(registry: ModelRegistry):
    registry.update_model("demo-image", setup_complete=True)
    assert registry.get_model_entry("demo-image")["setup_complete"] is True


def test_scan_ignores_folders_without_config(models_dir: Path):
    (models_dir / "not-a-model").mkdir()
    (models_dir / "not-a-model" / "readme.txt").write_text("hi", encoding="utf-8")
    registry = ModelRegistry(models_dir)
    report = registry.scan_and_sync()
    assert "not-a-model" in report["missing_config"]
    assert not registry.exists("not-a-model")


def test_scan_detects_new_folder(models_dir: Path):
    new = models_dir / "brand-new"
    new.mkdir()
    (new / "config.yaml").write_text(
        yaml.safe_dump({"type": "tts", "backend": "local", "tags": ["x"]}), encoding="utf-8"
    )
    registry = ModelRegistry(models_dir)
    report = registry.scan_and_sync()
    assert "brand-new" in report["added"]
    assert registry.get_model_entry("brand-new")["type"] == "tts"


def test_atomic_write_no_corruption_on_partial(models_dir: Path):
    """A leftover .tmp file must not corrupt the real registry."""
    registry = ModelRegistry(models_dir)
    tmp = registry.registry_path.with_suffix(registry.registry_path.suffix + ".tmp")
    tmp.write_text("this: is: broken: yaml: [", encoding="utf-8")
    # Real registry still loads fine and a normal write succeeds.
    registry.add_model("post-crash", type="tts", backend="local")
    assert registry.exists("post-crash")
    loaded = yaml.safe_load(registry.registry_path.read_text(encoding="utf-8"))
    assert "post-crash" in loaded["models"]
