"""Shared fixtures: a real temp MODELS_DIR with a couple of model folders."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from modelrack.process_manager import ProcessManager
from modelrack.registry import ModelRegistry
from modelrack.resolver import ModelResolver

# A diffusers-style image model (weights = directory).
DIFFUSERS_CONFIG = {
    "model_id": "demo-image",
    "display_name": "Demo Image",
    "type": "image_generation",
    "backend": "local",
    "weights": {"main": "weights", "vae": "weights/vae.safetensors"},
    "hardware": {"dtype": "bfloat16", "device": "cuda", "min_vram_gb": 8},
    "server": {"port": 7801, "host": "127.0.0.1", "startup_timeout_sec": 30},
    "environment": {
        "python_version": "3.11",
        "requirements_file": "requirements.txt",
        "venv_path": ".venv",
    },
    "defaults": {"num_inference_steps": 50, "guidance_scale": 7.5, "seed": -1},
    "param_schema": {
        "num_inference_steps": {"type": "int", "min": 1, "max": 100, "label": "Steps"},
        "guidance_scale": {"type": "float", "min": 1.0, "max": 20.0, "label": "CFG"},
        "sampler": {"type": "str", "options": ["euler", "ddim"], "label": "Sampler"},
    },
    "load_hints": {"framework": "diffusers", "pipeline_class": "DemoPipeline"},
    "serving": {"engine": "diffusers"},
    "tags": ["image", "diffusion"],
}

# An LLM served via vLLM.
LLM_CONFIG = {
    "model_id": "demo-llm",
    "display_name": "Demo LLM",
    "type": "language",
    "backend": "local",
    "weights": {"main": "weights"},
    "hardware": {"dtype": "bfloat16", "device": "cuda"},
    "server": {"port": 7809, "host": "127.0.0.1", "startup_timeout_sec": 30},
    "environment": {
        "python_version": "3.11",
        "requirements_file": "requirements.txt",
        "venv_path": ".venv",
    },
    "defaults": {"temperature": 0.7, "top_p": 0.8, "max_tokens": 2048},
    "param_schema": {
        "temperature": {"type": "float", "min": 0.0, "max": 2.0, "label": "Temp"},
        "max_tokens": {"type": "int", "min": 1, "max": 8192, "label": "Max Tokens"},
    },
    "serving": {"engine": "vllm", "tensor_parallel_size": 1},
    "tags": ["language", "llm"],
}


def _write_model(models_dir: Path, config: dict, with_weights: bool = True) -> Path:
    model_id = config["model_id"]
    model_dir = models_dir / model_id
    (model_dir / "weights").mkdir(parents=True, exist_ok=True)
    (model_dir / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (model_dir / "requirements.txt").write_text("torch\n", encoding="utf-8")
    if with_weights:
        # weights/ dir itself is the "main" weight for diffusers-style models.
        (model_dir / "weights" / "model.safetensors").write_bytes(b"\x00")
    return model_dir


@pytest.fixture
def models_dir(tmp_path: Path) -> Path:
    """A temp MODELS_DIR containing demo-image and demo-llm, plus a registry.yaml."""
    root = tmp_path / "models"
    root.mkdir()
    _write_model(root, DIFFUSERS_CONFIG)
    _write_model(root, LLM_CONFIG)

    registry = ModelRegistry(root)
    registry.scan_and_sync()
    return root


@pytest.fixture
def registry(models_dir: Path) -> ModelRegistry:
    return ModelRegistry(models_dir)


@pytest.fixture
def resolver(models_dir: Path) -> ModelResolver:
    return ModelResolver(models_dir)


@pytest.fixture
def process_manager(models_dir: Path, tmp_path: Path) -> ProcessManager:
    state_file = tmp_path / "state" / "processes.json"
    return ProcessManager(models_dir, state_file=state_file)
