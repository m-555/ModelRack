"""Tests for ModelResolver (3-layer merge, weight resolution, app overrides)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from modelrack.exceptions import (
    AppOverridesNotFoundError,
    ParamValidationError,
    WeightNotFoundError,
)
from modelrack.registry import ModelRegistry
from modelrack.resolver import ModelResolver


def test_resolve_absolute_weight_paths(resolver: ModelResolver, models_dir: Path):
    resolved = resolver.resolve("demo-image")
    assert resolved.weight_paths is not None
    main = resolved.weight_paths["main"]
    assert main.is_absolute()
    assert main == (models_dir / "demo-image" / "weights").resolve()


def test_defaults_present_without_overrides(resolver: ModelResolver):
    resolved = resolver.resolve("demo-image")
    assert resolved.merged_config["defaults"]["num_inference_steps"] == 50


def test_three_layer_merge_runtime_wins(resolver: ModelResolver):
    resolved = resolver.resolve(
        "demo-image",
        app_overrides={"defaults": {"num_inference_steps": 30, "guidance_scale": 5.0}},
        runtime_params={"num_inference_steps": 20},
    )
    d = resolved.merged_config["defaults"]
    assert d["num_inference_steps"] == 20  # runtime beats app
    assert d["guidance_scale"] == 5.0  # app beats base
    assert d["seed"] == -1  # base preserved


def test_deep_merge_on_nested_section(resolver: ModelResolver):
    resolved = resolver.resolve("demo-image", app_overrides={"hardware": {"dtype": "float16"}})
    assert resolved.merged_config["hardware"]["dtype"] == "float16"
    assert resolved.merged_config["hardware"]["device"] == "cuda"  # untouched


def test_missing_optional_weight_is_skipped(resolver: ModelResolver):
    # demo-image declares an optional 'vae' weight that does not exist on disk.
    resolved = resolver.resolve("demo-image")
    assert resolved.weight_paths is not None
    assert "vae" not in resolved.weight_paths
    assert "main" in resolved.weight_paths


def test_missing_main_weight_raises(models_dir: Path):
    # Remove the weights directory so 'main' cannot resolve.
    import shutil

    shutil.rmtree(models_dir / "demo-image" / "weights")
    resolver = ModelResolver(models_dir)
    with pytest.raises(WeightNotFoundError):
        resolver.resolve("demo-image")


def test_runtime_param_out_of_range_raises(resolver: ModelResolver):
    with pytest.raises(ParamValidationError):
        resolver.resolve("demo-image", runtime_params={"num_inference_steps": 999})


def test_runtime_param_invalid_option_raises(resolver: ModelResolver):
    with pytest.raises(ParamValidationError):
        resolver.resolve("demo-image", runtime_params={"sampler": "not-a-sampler"})


def test_get_param_schema(resolver: ModelResolver):
    schema = resolver.get_param_schema("demo-image")
    assert "num_inference_steps" in schema
    assert schema["num_inference_steps"]["max"] == 100


def test_resolve_from_app_file(resolver: ModelResolver, tmp_path: Path):
    overrides = tmp_path / "overrides.yaml"
    overrides.write_text(
        yaml.safe_dump({"models": {"demo-image": {"defaults": {"num_inference_steps": 12}}}}),
        encoding="utf-8",
    )
    resolved = resolver.resolve_from_app("demo-image", overrides)
    assert resolved.merged_config["defaults"]["num_inference_steps"] == 12


def test_resolve_from_app_missing_file_raises(resolver: ModelResolver, tmp_path: Path):
    with pytest.raises(AppOverridesNotFoundError):
        resolver.resolve_from_app("demo-image", tmp_path / "nope.yaml")


def test_list_available_skips_broken(models_dir: Path):
    # Corrupt one config; list_available should skip it, not raise.
    (models_dir / "demo-llm" / "config.yaml").write_text("{[", encoding="utf-8")
    resolver = ModelResolver(models_dir)
    ids = {r.model_id for r in resolver.list_available()}
    assert "demo-image" in ids


def test_api_model_has_no_weights(models_dir: Path):
    registry = ModelRegistry(models_dir)
    registry.add_model(
        "claude",
        type="language",
        backend="api",
        provider="anthropic",
        api_model_id="claude-opus-4-8",
    )
    resolver = ModelResolver(models_dir, registry)
    resolved = resolver.resolve("claude")
    assert resolved.is_api
    assert resolved.weight_paths is None
    assert resolved.provider == "anthropic"
    assert resolved.api_model_id == "claude-opus-4-8"


def test_server_url_built_from_port(resolver: ModelResolver):
    resolved = resolver.resolve("demo-image")
    assert resolved.server_port == 7801
    assert resolved.server_url == "http://127.0.0.1:7801"
