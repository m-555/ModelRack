"""Tests for ConfigValidator."""

from __future__ import annotations

import copy

from tests.conftest import DIFFUSERS_CONFIG

from modelrack.resolver import ModelResolver
from modelrack.validator import ConfigValidator


def test_valid_config_passes():
    result = ConfigValidator().validate_model_config(DIFFUSERS_CONFIG)
    assert result.valid, result.errors


def test_missing_required_fields_fail():
    cfg = copy.deepcopy(DIFFUSERS_CONFIG)
    del cfg["type"]
    result = ConfigValidator().validate_model_config(cfg)
    assert not result.valid
    assert any("type" in e for e in result.errors)


def test_bad_port_fails():
    cfg = copy.deepcopy(DIFFUSERS_CONFIG)
    cfg["server"]["port"] = 99999
    result = ConfigValidator().validate_model_config(cfg)
    assert not result.valid


def test_unknown_type_warns_not_errors():
    cfg = copy.deepcopy(DIFFUSERS_CONFIG)
    cfg["type"] = "some_future_kind"
    result = ConfigValidator().validate_model_config(cfg)
    assert result.valid
    assert any("Unknown model type" in w for w in result.warnings)


def test_runtime_params_collect_all_violations():
    schema = DIFFUSERS_CONFIG["param_schema"]
    result = ConfigValidator().validate_runtime_params(
        {"num_inference_steps": 999, "guidance_scale": 0.0, "sampler": "bogus"}, schema
    )
    assert not result.valid
    # one over-max int, one under-min float, one bad option
    assert len(result.errors) >= 3


def test_runtime_param_type_mismatch():
    schema = {"steps": {"type": "int", "min": 1, "max": 10, "label": "S"}}
    result = ConfigValidator().validate_runtime_params({"steps": "five"}, schema)
    assert not result.valid


def test_weight_paths_missing_main_errors(models_dir):
    import shutil

    shutil.rmtree(models_dir / "demo-image" / "weights")
    (models_dir / "demo-image" / "weights").mkdir()  # dir exists but is empty
    # Point main at a non-existent file to force the error path.
    cfg_path = models_dir / "demo-image" / "config.yaml"
    import yaml

    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["weights"]["main"] = "weights/missing.safetensors"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    resolver = ModelResolver(models_dir)
    # resolve() would raise; build the ResolvedModel via API model path is overkill,
    # so validate directly on a resolved object that skips the resolver's own check.
    resolved = resolver.resolve("demo-llm")  # a valid one
    resolved.merged_config["weights"] = {"main": "weights/missing.safetensors"}
    result = ConfigValidator().validate_weight_paths(resolved)
    assert not result.valid


def test_venv_missing_warns(resolver: ModelResolver):
    resolved = resolver.resolve("demo-image")
    result = ConfigValidator().validate_venv(resolved)
    assert result.valid  # warning, not error
    assert any(".venv" in w for w in result.warnings)


def test_validate_all_merges(resolver: ModelResolver):
    resolved = resolver.resolve("demo-image")
    result = ConfigValidator().validate_all(resolved)
    # main weight (the weights/ dir) exists, so no errors; venv warning present.
    assert result.valid
    assert result.warnings
